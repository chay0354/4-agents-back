"""
Test script to simulate the system and verify hard stop functionality
"""
import asyncio
import httpx
import json
import time

BACKEND_URL = "http://localhost:8000"

async def test_hard_stop():
    """Test that hard stop actually stops the analysis"""
    print("=" * 60)
    print("Testing Hard Stop Functionality")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Start analysis
        print("\n1. Starting analysis...")
        problem = "What is the capital of France?"
        
        try:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/analyze",
                json={"problem": problem},
                headers={"Content-Type": "application/json"},
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    print(f"ERROR: Failed to start analysis: {response.status_code}")
                    return
                
                print("OK: Analysis started, streaming responses...")
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                
                # Step 2: Read streaming responses
                agent_updates = {}
                stopped = False
                stop_triggered = False
                chunks_received = 0
                
                async def read_stream():
                    nonlocal stopped, stop_triggered, chunks_received
                    buffer = ""
                    
                    try:
                        async for chunk in response.aiter_text():
                            chunks_received += 1
                            if chunks_received == 1:
                                print(f"First chunk received (length: {len(chunk)})")
                            if stopped:
                                break
                            buffer += chunk
                            lines = buffer.split('\n')
                            buffer = lines.pop() if lines else ""
                            
                            for line in lines:
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        agent = data.get('agent', 'unknown')
                                        status = data.get('status', 'unknown')
                                        
                                        print(f"  [Received] agent={agent}, status={status}")
                                        
                                        # Track agent updates
                                        if agent in ['analysis', 'research', 'critic', 'monitor']:
                                            agent_updates[agent] = {
                                                'status': status,
                                                'has_response': 'response' in data and data['response'] is not None,
                                                'response_length': len(data.get('response', '')) if data.get('response') else 0
                                            }
                                            
                                            # Show response immediately when agent completes
                                            if status == 'complete' and data.get('response'):
                                                response_preview = data['response'][:100] + "..." if len(data.get('response', '')) > 100 else data.get('response', '')
                                                print(f"  [COMPLETE] {agent.capitalize()} Agent - Response length: {len(data['response'])}")
                                                print(f"     Preview: {response_preview}")
                                        
                                        # Check for stop
                                        if agent == 'system' and status == 'stopped':
                                            print(f"\n  [STOPPED] {data.get('message', 'Analysis stopped')}")
                                            stopped = True
                                            return
                                            
                                    except json.JSONDecodeError as e:
                                        print(f"  [WARNING] Error parsing JSON: {e}")
                    except Exception as e:
                        if not stopped:
                            print(f"  [ERROR] Stream error: {e}")
                
                # Step 3: Wait a bit, then trigger hard stop
                async def trigger_stop():
                    nonlocal stop_triggered
                    # Wait 3 seconds to let analysis agent start
                    await asyncio.sleep(3)
                    
                    print("\n2. Triggering HARD STOP...")
                    # Use a separate client for the stop command
                    async with httpx.AsyncClient(timeout=10.0) as stop_client:
                        try:
                            stop_response = await stop_client.post(f"{BACKEND_URL}/kernel/stop")
                            
                            if stop_response.status_code == 200:
                                print("OK: Hard stop command sent successfully")
                                stop_triggered = True
                            else:
                                print(f"ERROR: Failed to send stop command: {stop_response.status_code}")
                        except Exception as e:
                            print(f"ERROR: Exception sending stop command: {e}")
                
                # Run both tasks concurrently
                stream_task = asyncio.create_task(read_stream())
                stop_task = asyncio.create_task(trigger_stop())
                
                # Wait for stream to finish or timeout
                try:
                    await asyncio.wait_for(stream_task, timeout=30.0)
                except asyncio.TimeoutError:
                    print("\n[TIMEOUT] Stream timeout (30s)")
                
                # Wait a bit more to see if stop happens
                await asyncio.sleep(2)
                
                # Step 4: Verify results
                print("\n" + "=" * 60)
                print("Test Results:")
                print("=" * 60)
                
                print(f"\nHard stop triggered: {stop_triggered}")
                print(f"Analysis stopped: {stopped}")
                print(f"Chunks received: {chunks_received}")
                print(f"\nAgent updates received:")
                if agent_updates:
                    for agent, info in agent_updates.items():
                        print(f"  - {agent}: status={info['status']}, has_response={info['has_response']}, length={info['response_length']}")
                else:
                    print("  (no agent updates received)")
                
                # Check kernel history
                print("\n3. Checking kernel history...")
                try:
                    history_response = await client.get(f"{BACKEND_URL}/kernel/history")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        print(f"OK: Stop events in history: {history.get('count', 0)}")
                        if history.get('history'):
                            for event in history['history'][-3:]:  # Show last 3 events
                                print(f"   - {event.get('timestamp', '')}: {event.get('action', '')} (agent: {event.get('stopped_agent', 'N/A')})")
                except Exception as e:
                    print(f"ERROR: Could not get kernel history: {e}")
                
                # Final verdict
                print("\n" + "=" * 60)
                if stop_triggered and stopped:
                    print("SUCCESS: Hard stop works correctly!")
                    print("   - Stop command was sent")
                    print("   - Analysis stopped as expected")
                elif stop_triggered and not stopped:
                    print("FAILURE: Hard stop was triggered but analysis didn't stop!")
                    print("   - Stop command was sent")
                    print("   - But analysis continued running")
                else:
                    print("INCOMPLETE: Could not verify stop functionality")
                print("=" * 60)
        except httpx.ConnectError:
            print("ERROR: Could not connect to backend. Make sure the server is running on http://localhost:8000")
            return

if __name__ == "__main__":
    print("Make sure the backend server is running on http://localhost:8000")
    print("Press Ctrl+C to cancel\n")
    time.sleep(2)
    
    try:
        asyncio.run(test_hard_stop())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
