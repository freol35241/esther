import queue
from time import monotonic
from types import MethodType

from streamz import Stream, Sink, combine_latest
from streamz.sources import from_q


@Stream.register_api()
class combine_latest_with_timeout(combine_latest):
    def __init__(self, *upstreams, timeout=None, **kwargs):
        self._timeout = timeout
        self._last_seen = [float("inf")] * len(upstreams)
        super().__init__(*upstreams, **kwargs)

    def update(self, x, who=None, metadata=None):
        now = monotonic()
        self._retain_refs(metadata)
        idx = self.upstreams.index(who)
        if self.metadata[idx]:
            self._release_refs(self.metadata[idx])
        self.metadata[idx] = metadata
        self._last_seen[idx] = now

        diffs = [now - last_seen for last_seen in self._last_seen]

        if self.missing and who in self.missing:
            self.missing.remove(who)

        self.last[idx] = x
        if not self.missing and who in self.emit_on and max(diffs) < self._timeout:
            tup = tuple(self.last)
            md = [m for ml in self.metadata for m in ml]
            return self._emit(tup, md)


@Stream.register_api()
class on_exception(Sink):
    def __init__(self, upstream: Stream, exception=Exception, **kwargs):
        super().__init__(upstream, **kwargs)

        original_upstream_update_method = upstream.update

        def _(upstream_self, x, who=None, metadata=None):
            try:
                return original_upstream_update_method(x, who, metadata)
            except exception as exc:
                # Pass down the branch started with this stream instead
                self._emit((x, exc), metadata)

        # Bind to upstream
        upstream.update = MethodType(_, upstream)

    def update(self, x, who=None, metadata=None):
        pass  # NO-OP


@Stream.register_api(staticmethod, attribute_name="from_secured_mqtt")
class from_mqtt(from_q):
    """Read from MQTT source
    See https://en.wikipedia.org/wiki/MQTT for a description of the protocol
    and its uses.
    See also ``sinks.to_mqtt``.
    Requires ``paho.mqtt``
    The outputs are ``paho.mqtt.client.MQTTMessage`` instances, which each have
    attributes timestamp, payload, topic, ...
    NB: paho.mqtt.python runs on its own thread in this implementation. We may
    wish to instead call client.loop() directly
    :param host: str
    :param port: int
    :param topic: str
        (May in the future support a list of topics)
    :param keepalive: int
        See mqtt docs - to keep the channel alive
    :param client_kwargs:
        Passed to the client's ``connect()`` method
    """

    def __init__(
        self,
        host,
        port,
        topic,
        username=None,
        password=None,
        keepalive=60,
        client_kwargs=None,
        **kwargs
    ):

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self.topic = topic
        self.client_kwargs = client_kwargs
        super().__init__(q=queue.Queue(), **kwargs)

    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        self.q.put(msg)

    async def run(self):
        import paho.mqtt.client as mqtt

        client = mqtt.Client()
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.username_pw_set(self.username, self.password)
        client.connect(
            self.host, self.port, self.keepalive, **(self.client_kwargs or {})
        )
        client.loop_start()
        await super().run()
        client.disconnect()


@Stream.register_api(attribute_name="to_secured_mqtt")
class to_mqtt(Sink):
    """
    Send data to MQTT broker
    See also ``sources.from_mqtt``.
    Requires ``paho.mqtt``
    :param host: str
    :param port: int
    :param topic: str
    :param keepalive: int
        See mqtt docs - to keep the channel alive
    :param client_kwargs:
        Passed to the client's ``connect()`` method
    """

    def __init__(
        self,
        upstream,
        host,
        port,
        topic,
        username=None,
        password=None,
        keepalive=60,
        client_kwargs=None,
        **kwargs
    ):
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client()
        self.client.username_pw_set(username, password)
        self.client.connect(host, port, keepalive, **(client_kwargs or {}))
        self.client.loop_start()

        self.topic = topic
        super().__init__(upstream, ensure_io_loop=True, **kwargs)

    def update(self, x, who=None, metadata=None):
        # TODO: wait on successful delivery
        if self.client.is_connected():
            self.client.publish(self.topic, x)

    def destroy(self):
        self.client.loop_stop()
        self.client.disconnect()
        super().destroy()
