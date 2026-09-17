import copy
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import h5py
import numpy as np
import pytest
from area_detector_handlers.handlers import BulkXSPRESS, Xspress3HDF5Handler
from bluesky import plans
from bluesky_tiled_plugins import TiledWriter
from event_model import DocumentNames
from ophyd import Component, Kind
from ophyd.areadetector import ADBase, Xspress3Detector
from ophyd.sim import NullStatus, make_fake_device
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.media_type_registration import default_serialization_registry
from tiled.server.app import build_app

from bluesky_tiled_plugins.exporters import json_seq_exporter
from bluesky_tiled_plugins.routers import validator

from nslsii.areadetector.xspress3 import (
    Xspress3HDF5Plugin,
    Xspress3Trigger,
    build_xspress3_class,
)
from nslsii.areadetector.xspress3_stream import Xspress3HDF5StreamPlugin


EXPECTED_DATA = np.arange(2 * 2 * 4096, dtype=np.uint32).reshape(2, 2, 4096)
DATASET_PATH = "/entry/instrument/detector/data"


@pytest.fixture
def tiled_client(tmp_path):
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    catalog = in_memory(
        specs=[{"name": "CatalogOfBlueskyRuns", "version": "3.0"}],
        writable_storage={
            "filesystem": str(asset_dir),
            "sql": f"duckdb:///{tmp_path}/tabular_data.db",
        },
        readable_storage=[str(asset_dir)],
    )
    serialization_registry = copy.copy(default_serialization_registry)
    serialization_registry.register("BlueskyRun", "application/json-seq", json_seq_exporter)
    app = build_app(
        catalog,
        serialization_registry=serialization_registry,
        include_routers=[validator.router],
    )
    context = Context.from_app(app)
    try:
        yield from_context(context), asset_dir
    finally:
        context.close()


class _SimulatedPluginMixin:
    def stage(self):
        self.stage_sigs[self.file_template] = "%s/%s_%6.6d.h5"
        signals = {getattr(self, signal) if isinstance(signal, str) else signal for signal in self.stage_sigs}
        signals.update(
            getattr(self, signal)
            for signal in ("array_counter", "file_path", "file_name", "file_number", "capture")
        )
        # Fake EPICS stage signals have no IOC readback to complete their normal set().
        for signal in signals:
            if hasattr(signal, "sim_put"):

                def set_signal(value, *args, signal=signal, **kwargs):
                    signal.sim_put(value)
                    return NullStatus()

                signal.set = set_signal

        staged_devices = super().stage()

        resource = getattr(self, "_resource", None)
        if resource is not None:
            file_path = Path(resource["root"]) / resource["resource_path"]
        else:
            stream_resources = [
                document
                for name, document in self._asset_docs_cache
                if name == DocumentNames.stream_resource.value
            ]
            uri = stream_resources[0]["uri"]
            file_path = Path(unquote(urlparse(uri).path))

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(file_path, "w") as file:
            file.create_dataset(DATASET_PATH, data=EXPECTED_DATA, chunks=(1, 2, 4096))

        return staged_devices


class SimulatedXspress3HDF5Plugin(_SimulatedPluginMixin, Xspress3HDF5Plugin):
    pass


class SimulatedXspress3HDF5StreamPlugin(_SimulatedPluginMixin, Xspress3HDF5StreamPlugin):
    pass


def _build_fake_detector(asset_dir, plugin_class):
    detector_class = build_xspress3_class(
        channel_numbers=(1, 2),
        mcaroi_numbers=(),
        image_data_key="image",
        xspress3_parent_classes=(Xspress3Detector, Xspress3Trigger),
        extra_class_members={
            "hdf5plugin": Component(
                plugin_class,
                "HDF1:",
                name="h5p",
                root_path=str(asset_dir),
                path_template=str(asset_dir),
                resource_kwargs={},
            )
        },
    )
    fake_detector_class = make_fake_device(detector_class)
    fake_detector_class.channel01.cls.__init__ = ADBase.__init__
    fake_detector_class.channel02.cls.__init__ = ADBase.__init__
    detector = fake_detector_class(prefix="Xsp3:", name="det")
    detector.hdf5plugin.array_size.depth.sim_put(1)
    detector.hdf5plugin.array_size.height.sim_put(1)
    detector.hdf5plugin.array_size.width.sim_put(1)
    detector.hdf5plugin.plugin_type.sim_put("NDFileHDF5")
    return detector


def test_xspress3_channel_stream_shapes(RE, tiled_client):
    client, asset_dir = tiled_client
    detector = _build_fake_detector(asset_dir, SimulatedXspress3HDF5StreamPlugin)
    detector.read_attrs = ["image", "channel01.image", "channel02.image"]
    detector.image.kind = Kind.normal

    production_trigger = detector.trigger

    def trigger_and_finish():
        status = production_trigger()
        detector.cam.acquire.put(0)
        return status

    detector.trigger = trigger_and_finish
    parent_key = detector.image.name
    channel_keys = [detector.channel01.image.name, detector.channel02.image.name]
    data_keys = [parent_key, *channel_keys]

    documents = []
    subscription = RE.subscribe(lambda name, document: documents.append((name, document)))
    try:
        writer = TiledWriter(client, validate=False)
        RE(plans.count([detector], 2), writer)
    finally:
        RE.unsubscribe(subscription)

    run = client.v3.values().last()
    stream = run["primary"]

    parent_node = stream[parent_key]
    assert parent_node.structure().shape == (2, 2, 4096)
    np.testing.assert_array_equal(parent_node.read(), EXPECTED_DATA)

    for channel_index, channel_key in enumerate(channel_keys):
        node = stream[channel_key]
        assert node.structure().shape == (2, 4096)
        actual = node.read()
        assert actual.shape == (2, 4096)
        np.testing.assert_array_equal(actual, EXPECTED_DATA[:, channel_index, :])

    stream_resources = {
        document["data_key"]: document
        for name, document in documents
        if name == DocumentNames.stream_resource.value
    }
    assert set(stream_resources) == set(data_keys)
    assert len(stream_resources) == 3
    parent_parameters = stream_resources[parent_key]["parameters"]
    assert "slice" not in parent_parameters
    assert tuple(parent_parameters["chunk_shape"]) == (1, 2, 4096)
    assert parent_parameters["dataset"] == DATASET_PATH
    assert parent_parameters["join_method"] == "stack"
    assert parent_parameters["spec"] == "XSP3"
    for channel_index, channel_key in enumerate(channel_keys):
        parameters = stream_resources[channel_key]["parameters"]
        assert stream_resources[channel_key]["mimetype"] == "application/x-hdf5"
        assert parameters["dataset"] == DATASET_PATH
        assert parameters["slice"] == f":,{channel_index},:"
        assert tuple(parameters["chunk_shape"]) == (1, 4096)
        assert parameters["join_method"] == "stack"
        assert parameters["spec"] == "XSP3"

    stream_datums = [document for name, document in documents if name == DocumentNames.stream_datum.value]
    assert len(stream_datums) == 6
    datum_counts = Counter(document["stream_resource"] for document in stream_datums)
    assert set(datum_counts) == {document["uid"] for document in stream_resources.values()}
    assert set(datum_counts.values()) == {2}

    assert (
        run.validate(
            fix_errors=False,
            try_reading=True,
            raise_on_error=True,
            write_notes=False,
        )
        is True
    )


def test_xspress3_legacy_asset_shapes(tmp_path):
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    detector = _build_fake_detector(asset_dir, SimulatedXspress3HDF5Plugin)
    detector.image.kind = Kind.normal
    detector.hdf5plugin.stage()
    try:
        detector.hdf5plugin.generate_datum(key=None, timestamp=0, datum_kwargs={"frame": 0})
        asset_docs = list(detector.hdf5plugin.collect_asset_docs())
        assert Counter(name for name, _ in asset_docs) == Counter({"resource": 2, "datum": 3})

        resources = {document["uid"]: document for name, document in asset_docs if name == "resource"}
        bulk_resource = next(document for document in resources.values() if document["spec"] == "XSP3_FLY")
        channel_resource = next(document for document in resources.values() if document["spec"] == "XSP3")
        bulk_datum = next(
            document
            for name, document in asset_docs
            if name == "datum" and document["resource"] == bulk_resource["uid"]
        )
        channel_datums = [
            document
            for name, document in asset_docs
            if name == "datum" and document["resource"] == channel_resource["uid"]
        ]
        assert len(channel_datums) == 2

        file_path = Path(bulk_resource["root"]) / bulk_resource["resource_path"]
        with BulkXSPRESS(str(file_path)) as handler:
            np.testing.assert_array_equal(handler(**bulk_datum["datum_kwargs"]), EXPECTED_DATA)

        with Xspress3HDF5Handler(str(file_path)) as handler:
            for datum in channel_datums:
                kwargs = datum["datum_kwargs"]
                np.testing.assert_array_equal(
                    handler(**kwargs), EXPECTED_DATA[kwargs["frame"], kwargs["channel"] - 1, :]
                )

        assert detector.image.describe()[detector.image.name]["external"] == "FILESTORE:"
        assert detector.image.describe()[detector.image.name]["shape"] == (4096,)
        for channel in detector.iterate_channels():
            reference = channel.get_external_file_ref()
            assert reference.describe()[reference.name]["shape"] == (4096,)
            assert reference.describe()[reference.name]["external"] == "FILESTORE:"
    finally:
        detector.hdf5plugin.unstage()
