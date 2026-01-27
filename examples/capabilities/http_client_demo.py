"""
Demonstrates HTTPClient capability for making API requests.

Shows how agents can interact with REST APIs using the HTTP client.
"""

import asyncio

from cogent.capabilities.http_client import HTTPClient


async def main():
    """Demonstrate HTTP client capability."""
    
    # Create HTTP client with production-ready settings
    client = HTTPClient(
        timeout=30,
        max_retries=3,
        default_headers={"User-Agent": "CogentBot/1.0"},
    )
    
    print("=" * 70)
    print("HTTPClient Capability Demo")
    print("=" * 70)
    print()
    
    # Test 1: GET request to public API
    print("Test 1: GET request to JSONPlaceholder API")
    print("-" * 70)
    response = await client.get("https://jsonplaceholder.typicode.com/posts/1")
    
    if response.success:
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response time: {response.response_time_ms:.0f}ms")
        print(f"\nResponse body:")
        data = response.json()
        print(f"  Title: {data['title']}")
        print(f"  Body: {data['body'][:100]}...")
    else:
        print(f"✗ Request failed: {response.error}")
    print()
    
    # Test 2: POST request with JSON body
    print("Test 2: POST request with JSON data")
    print("-" * 70)
    post_data = {
        "title": "Test Post",
        "body": "This is a test post from Cogent HTTPClient",
        "userId": 1,
    }
    
    response = await client.post(
        "https://jsonplaceholder.typicode.com/posts",
        body=post_data,
    )
    
    if response.success:
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response time: {response.response_time_ms:.0f}ms")
        print(f"\nCreated resource:")
        data = response.json()
        print(f"  ID: {data.get('id')}")
        print(f"  Title: {data.get('title')}")
    else:
        print(f"✗ Request failed: {response.error}")
    print()
    
    # Test 3: GET request with query parameters
    print("Test 3: GET request with query parameters")
    print("-" * 70)
    response = await client.get(
        "https://jsonplaceholder.typicode.com/posts",
        params={"userId": "1"},
    )
    
    if response.success:
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response time: {response.response_time_ms:.0f}ms")
        data = response.json()
        print(f"\nFound {len(data)} posts for user 1")
        print(f"  First post: {data[0]['title']}")
    else:
        print(f"✗ Request failed: {response.error}")
    print()
    
    # Test 4: PUT request
    print("Test 4: PUT request to update resource")
    print("-" * 70)
    update_data = {
        "id": 1,
        "title": "Updated Title",
        "body": "Updated body content",
        "userId": 1,
    }
    
    response = await client.put(
        "https://jsonplaceholder.typicode.com/posts/1",
        body=update_data,
    )
    
    if response.success:
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response time: {response.response_time_ms:.0f}ms")
        print("✓ Resource updated successfully")
    else:
        print(f"✗ Request failed: {response.error}")
    print()
    
    # Test 5: DELETE request
    print("Test 5: DELETE request")
    print("-" * 70)
    response = await client.delete("https://jsonplaceholder.typicode.com/posts/1")
    
    if response.success:
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response time: {response.response_time_ms:.0f}ms")
        print("✓ Resource deleted successfully")
    else:
        print(f"✗ Request failed: {response.error}")
    print()
    
    # Test 6: Error handling - timeout
    print("Test 6: Error handling - Invalid URL")
    print("-" * 70)
    response = await client.get("https://this-domain-definitely-does-not-exist-12345.com")
    
    if not response.success:
        print(f"✗ Request failed as expected")
        print(f"  Error: {response.error}")
    print()
    
    # Show request history
    print("=" * 70)
    print("Request History")
    print("=" * 70)
    for i, req in enumerate(client.history, 1):
        status = "✓" if req.success else "✗"
        print(f"{i}. {status} {req.method} {req.url} - {req.status_code} ({req.response_time_ms:.0f}ms)")
    print()
    


if __name__ == "__main__":
    asyncio.run(main())
