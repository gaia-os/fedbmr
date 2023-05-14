import os
import json

from google.cloud import pubsub_v1
from typing import Optional, List


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../erudite-flag-364413-428e8c5a8dbb.json"

publisher_options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
# Sending messages to the same region ensures they are received in order
# even when multiple publishers are used.
client_options = {"api_endpoint": "us-east1-pubsub.googleapis.com:443"}

def pub(data: bytes, project_id: str, topic_id: str, ordering_key: Optional[str] = 'A') -> str:
    """Sends a message to Pub/Sub topic"""
    publisher = pubsub_v1.PublisherClient(
        publisher_options=publisher_options, client_options=client_options
    )

    topic_path = publisher.topic_path(project_id, topic_id)

    future = publisher.publish(topic_path, data, ordering_key=ordering_key)

    return future.result()


def sub(project_id: str, subscription_id: str, timeout: Optional[float] = None) -> List:
    """Receives messages from a Pub/Sub subscription."""
    # Initialize a Subscriber client
    subscriber_client = pubsub_v1.SubscriberClient()
    # Create a fully qualified identifier in the form of
    # `projects/{project_id}/subscriptions/{subscription_id}`
    subscription_path = subscriber_client.subscription_path(project_id, subscription_id)

    data = []
    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        print(f"Received {message}.")
        # Acknowledge the message. Unack'ed messages will be redelivered.
        message.ack()
        data.append(json.loads(message.data.decode()))
        print(f"Acknowledged {message.message_id}.")

    streaming_pull_future = subscriber_client.subscribe(
        subscription_path, callback=callback
    )
    print(f"Listening for messages on {subscription_path}..\n")

    try:
        # Calling result() on StreamingPullFuture keeps the main thread from
        # exiting while messages get processed in the callbacks.
        streaming_pull_future.result(timeout=timeout)
    except:  # noqa
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.

    subscriber_client.close()

    return data