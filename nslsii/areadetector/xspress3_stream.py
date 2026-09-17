import datetime
import logging
from collections import deque
from pathlib import Path
from uuid import uuid4

from event_model import compose_stream_resource
from ophyd import Kind
from ophyd.areadetector.plugins import HDF5Plugin_V34 as HDF5Plugin

from .xspress3 import Xspress3HDF5Plugin


logger = logging.getLogger(__name__)


class Xspress3HDF5StreamPlugin(Xspress3HDF5Plugin):
    """Xspress3 HDF5 plugin that emits native stream asset documents.

    This is the stream-asset counterpart to ``Xspress3HDF5Plugin``. The
    legacy plugin remains responsible for Resource/Datum documents; this
    class emits only StreamResource/StreamDatum documents.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_datum_composers = {}

    def _configure_references(self):
        parent_reference = self.parent.get_external_file_ref()
        channels = tuple(self.parent.iterate_channels())
        channel_references = tuple(channel.get_external_file_ref() for channel in channels)

        if parent_reference is not None and channel_references:
            channel_reference = channel_references[0]
            parent_reference.shape = (len(channel_references), *channel_reference.shape)
            parent_reference.dims = ("channel", *channel_reference.dims)

        for reference in (parent_reference, *channel_references):
            if reference is not None:
                reference.external = "STREAM:"
        return parent_reference, channels

    def _compose_stream_resource(self, *, full_file_path, reference, channel_number=None):
        parameters = {
            **self.resource_kwargs,
            "dataset": "/entry/instrument/detector/data",
            "chunk_shape": (1, *reference.shape),
            "join_method": "stack",
            "spec": self.spec,
        }
        parameters.pop("slice", None)
        if channel_number is not None:
            parameters["slice"] = f":,{channel_number - 1},:"
        return compose_stream_resource(
            mimetype="application/x-hdf5",
            uri=full_file_path.absolute().as_uri(),
            data_key=reference.name,
            parameters=parameters,
        )

    def stage(self):
        parent_reference, channels = self._configure_references()
        logger.debug("staging '%s' of '%s'", self.name, self.parent.name)

        # Bypass the legacy asset-document implementation in the immediate
        # parent while retaining its file-plugin configuration and cleanup.
        staged_devices = HDF5Plugin.stage(self)

        self.array_counter.set(0).wait()

        the_full_data_dir_path = self._build_data_dir_path(
            the_datetime=datetime.datetime.now(),
            root_path=self.root_path.get(),
            path_template=self.path_template.get(),
        )
        self.file_path.set(the_full_data_dir_path).wait()
        self.file_name.set("-".join(str(uuid4()).split("-")[:-1])).wait()
        self.file_number.set(0).wait()

        file_path = self.file_path.get()
        file_name = self.file_name.get()
        file_number = self.file_number.get()
        full_file_path = Path(self.stage_sigs[self.file_template] % (file_path, file_name, file_number))

        self._asset_docs_cache = deque()
        self._stream_datum_composers = {}

        if parent_reference is not None and parent_reference.kind & Kind.normal:
            stream_resource, stream_datum_composer = self._compose_stream_resource(
                full_file_path=full_file_path,
                reference=parent_reference,
            )
            self._stream_datum_composers[parent_reference.name] = stream_datum_composer
            self._asset_docs_cache.append(("stream_resource", stream_resource))

        for channel in channels:
            channel_reference = channel.get_external_file_ref()
            if channel_reference is None or not channel_reference.kind & Kind.normal:
                continue
            stream_resource, stream_datum_composer = self._compose_stream_resource(
                full_file_path=full_file_path,
                reference=channel_reference,
                channel_number=channel.channel_number,
            )
            self._stream_datum_composers[channel_reference.name] = stream_datum_composer
            self._asset_docs_cache.append(("stream_resource", stream_resource))

        self.capture.set(1).wait()
        return staged_devices

    def generate_datum(self, key, timestamp, datum_kwargs):
        if key is not None:
            raise ValueError(f"'key' must be None but key='{key}'")

        try:
            frame = datum_kwargs["frame"]
        except KeyError as exc:
            raise ValueError("'frame' is required in datum_kwargs for Xspress3 stream data") from exc

        parent_reference = self.parent.get_external_file_ref()
        if parent_reference is not None and parent_reference.kind & Kind.normal:
            stream_datum = self._stream_datum_composers[parent_reference.name](
                indices={"start": frame, "stop": frame + 1}
            )
            self._asset_docs_cache.append(("stream_datum", stream_datum))
            parent_reference.put(stream_datum["uid"])

        for channel in self.parent.iterate_channels():
            channel_reference = channel.get_external_file_ref()
            if channel_reference is None or not channel_reference.kind & Kind.normal:
                continue
            stream_datum = self._stream_datum_composers[channel_reference.name](
                indices={"start": frame, "stop": frame + 1}
            )
            self._asset_docs_cache.append(("stream_datum", stream_datum))
            channel_reference.put(stream_datum["uid"])
