import os
import json

from google.cloud import pubsub_v1
from typing import Optional, List

PublisherClient = pubsub_v1.PublisherClient
SubscriberClient = pubsub_v1.SubscriberClient

publisher_options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
# Sending messages to the same region ensures they are received in order
# even when multiple publishers are used.
client_options = {"api_endpoint": "us-east1-pubsub.googleapis.com:443"}

def publisher() -> PublisherClient:
    return pubsub_v1.PublisherClient(
                publisher_options=publisher_options, 
                client_options=client_options
            )

def pub(publisher: PublisherClient, data: bytes, topic_path: str, ordering_key: Optional[str] = 'A') -> str:
    """Sends a message to Pub/Sub topic"""

    future = publisher.publish(topic_path, data, ordering_key=ordering_key)

    return future.result()

def sub(subscriber, subscription_path: str, timeout: Optional[float] = None, verbose: Optional[bool] = False) -> List:
    """Receives messages from a Pub/Sub subscription."""
    
    data = []
    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        # Acknowledge the message. Unack'ed messages will be redelivered.
        message.ack()
        data.append( message.data )
        if verbose:
            print(f"Received {message}.")
            print(f"Acknowledged {message.message_id}.")

    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback
    )
    
    try:
        # Calling result() on StreamingPullFuture keeps the main thread from
        # exiting while messages get processed in the callbacks.
        streaming_pull_future.result(timeout=timeout)
    except:  # noqa
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.

    subscriber.close()

    return data