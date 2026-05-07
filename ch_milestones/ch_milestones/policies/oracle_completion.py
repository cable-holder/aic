from threading import Event

from std_msgs.msg import String


class InsertionCompletionMonitor:
    def __init__(self, node, topic):
        self.node = node
        self.message = None
        self.event = Event()
        self.subscription = node.create_subscription(String, topic, self.on_event, 10)

    def on_event(self, msg):
        self.message = msg.data
        self.event.set()

    def completed(self):
        return self.event.is_set()

    def close(self):
        self.node.destroy_subscription(self.subscription)
