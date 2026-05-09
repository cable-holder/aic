from threading import Event

from std_msgs.msg import String


class InsertionCompletion:
    def __init__(self, node, topic):
        self.event = Event()
        self.subscription = node.create_subscription(String, topic, self.on_event, 10)

    def on_event(self, msg):
        self.event.set()

    def completed(self):
        return self.event.is_set()

    def close(self, node):
        node.destroy_subscription(self.subscription)
