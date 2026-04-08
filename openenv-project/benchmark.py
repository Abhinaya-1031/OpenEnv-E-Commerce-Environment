import subprocess
import json
import os
import sys
import argparse

# Tasks to Benchmark
ALL_TASKS = ["easy", "medium", "hard"]

def run_task(task_id):
    """Executes inference.py for a specific task and parses the output."""
    print(f"--- Running Benchmark: {task_id.upper()} Task ---")
    
    # Use the same python executable as the current one
    cmd = [sys.executable, "inference.py", "--task", task_id]
    
    # Run process and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error executing task {task_id}: {result.stderr}")
        return None

    # Parse the [END] line for metrics
    lines = result.stdout.splitlines()
    end_line = next((l for l in reversed(lines) if l.startswith("[END]")), None)
    
    metrics = {}
    if end_line:
        parts = end_line.split(" ")
        for part in parts:
            if "=" in part:
                k, v = part.split("=")
                metrics[k] = v
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="OpenEnv Triage Benchmark Runner")
    parser.add_argument(
        "--task", 
        nargs="+", 
        default=["all"], 
        help="Specific task(s) to run (easy, medium, hard, or all)."
    )
    args = parser.parse_args()

    # Determine tasks to run
    tasks_to_run = ALL_TASKS if "all" in args.task else [t.lower() for t in args.task]
    
    # Validate tasks
    for t in tasks_to_run:
        if t not in ALL_TASKS:
            print(f"Error: Unknown task '{t}'. Available tasks: easy, medium, hard, all")
            sys.exit(1)

    print("="*60)
    print(" OPENENV TRIAGE - SUBMISSION SCORE CARD ")
    print("="*60)
    
    results = []
    for task in tasks_to_run:
        m = run_task(task)
        if m:
            results.append({
                "task": task.upper(),
                "status": "PASS" if m.get("success") == "true" else "FAIL",
                "steps": m.get("steps", "0"),
                "total_reward": m.get("total_reward", "0.00")
            })
    
    # Print Final Table
    print("\n" + "="*60)
    print(f"{'TASK':<15} | {'STATUS':<10} | {'STEPS':<10} | {'TOTAL REWARD':<15}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['task']:<15} | {r['status']:<10} | {r['steps']:<10} | {r['total_reward']:<15}")
    
    print("="*60)
    print(" Benchmark Complete. Ready for Final Deployment. ")
    print("="*60)

if __name__ == "__main__":
    main()
