import asyncio
import os
import sys
from fastmcp import Client

# Default arguments to use as a baseline
DEFAULT_ARGS = {
    "date": "2024-06-20",
    "monthly_log_csv": "/home/arnav0103/github/SmartFolio/logs/monthly/2024-12/final_test_weights_20251128_015405.csv",
    "model_path": "/home/arnav0103/github/SmartFolio/checkpoints/ppo_hgat_custom_20251128_015405.zip",
    "market": "custom",
    "data_root": "dataset_default",
    "top_k": 5,
    "lookback_days": 60,
    "monthly_run_id": None,
    "output_dir": "explainability_results",
    "llm": True,
    "llm_model": "gemini-2.0-flash"
}

def get_user_input(current_args):
    """Prompts the user to override default arguments."""
    print("\n--- Configure SmartFolio XAI Run ---")
    print("Press Enter to keep [current value]. Type 'run' to execute immediately. Type 'exit' to quit.\n")
    
    args = current_args.copy()
    
    # Quick prompt for action
    action = input(f"Ready to run with date={args['date']}? (Press Enter to run, 'c' to configure, 'q' to quit): ").strip().lower()
    if action == 'q':
        return None
    if action != 'c':
        return args

    # Detailed configuration
    for key, default_val in args.items():
        prompt = f"{key} [{default_val}]: "
        user_val = input(prompt).strip()
        
        if user_val:
            if user_val.lower() == 'exit': return None
            
            # Type conversion
            if isinstance(default_val, bool):
                if user_val.lower() in ('y', 'yes', 'true', 't', '1'):
                    args[key] = True
                elif user_val.lower() in ('n', 'no', 'false', 'f', '0'):
                    args[key] = False
            elif isinstance(default_val, int):
                try:
                    args[key] = int(user_val)
                except ValueError:
                    print(f"Invalid integer for {key}, keeping default.")
            elif default_val is None:
                if user_val.lower() == "none":
                    args[key] = None
                else:
                    args[key] = user_val
            else:
                args[key] = user_val
    
    return args

async def run():
    print("🚀 Connecting to SmartFolio MCP Server (Streamable HTTP)...")
    print("   Ensure 'python3 start_mcp.py' is running in another terminal!")
    
    # Connect to the running server via FastMCP Client (Streamable HTTP)
    # The Pathway MCP server uses streamable-http transport on /mcp/ by default
    client = Client("http://localhost:9123/mcp/")

    async with client:
        print("✅ Server Connected!")
        
        current_args = DEFAULT_ARGS.copy()
        
        while True:
            run_args = get_user_input(current_args)
            if run_args is None:
                print("Exiting...")
                break
            
            # Update defaults for next run
            current_args = run_args
            
            print(f"\n🛠️  Executing 'run_xai_orchestrator' for {run_args['date']}...")
            
            try:
                # Call the tool
                result = await client.call_tool("run_xai_orchestrator", arguments=run_args)
                
                print("\n✅ Tool Execution Successful!")
                print("-" * 40)
                # FastMCP returns a list of Content objects or similar
                if hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                        else:
                            print(content)
                else:
                    # Fallback if result is just text or list
                    print(result)
                print("-" * 40)
                    
            except Exception as e:
                print(f"\n❌ Tool Execution Failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
