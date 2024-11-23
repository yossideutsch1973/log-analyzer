import ollama
import chromadb
import os
from tqdm import tqdm
import psutil
import torch
import time
import signal
import sys

def read_log_file(file_path):
    """Read and split the log file into chunks of reasonable size"""
    with open(file_path, 'r') as file:
        # Read the entire file
        content = file.read()
        # Split into chunks of roughly equal size (by lines)
        # This is a simple split - you might want to adjust chunk size
        chunks = content.split('\n')
        # Group chunks into larger segments (e.g., 10 lines per segment)
        return ['\n'.join(chunks[i:i+10]) for i in range(0, len(chunks), 10)]

def get_system_info():
    """Get system resource information"""
    process = psutil.Process()
    
    # Initialize GPU variables
    gpu_available = False
    gpu_name = "No GPU"
    gpu_info = {}

    # Check system CUDA setup
    print("\nCUDA Environment:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    
    # Try to get CUDA device count directly
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name,memory.total,memory.free', '--format=csv,noheader']).decode()
        print(f"\nNVIDIA-SMI GPU Info:\n{nvidia_smi}")
        
        # Force PyTorch to reinitialize CUDA
        if hasattr(torch.cuda, 'is_initialized'):
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
        # Check CUDA availability after reset
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"PyTorch detected CUDA devices: {device_count}")
            
            if device_count > 0:
                # Try to force device initialization
                torch.cuda.init()
                current_device = torch.cuda.current_device()
                torch.cuda.device(current_device)
                
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(current_device)
                
                # Get detailed GPU information
                device_props = torch.cuda.get_device_properties(current_device)
            # Get CUDA devices info
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
            
            gpu_info = {
                'cuda_version': torch.version.cuda,
                'device_name': gpu_name,
                'total_memory': f"{device_props.total_memory / 1024**3:.2f}GB",
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multi_processor_count': device_props.multi_processor_count,
                'cuda_devices': cuda_devices
            }
            
            print("\nGPU Information:")
            for key, value in gpu_info.items():
                print(f"{key}: {value}")
            
            # Warm up GPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"\nGPU detection error: {e}")
        print("Detailed error info:")
        import traceback
        traceback.print_exc()
        print("\nFalling back to CPU-only mode")
        gpu_info = {'error': str(e)}
        
        # Reset CUDA state
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    return {
        'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
        'cpu_percent': process.cpu_percent(),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name
    }

def signal_handler(signum, frame):
    print("\nGraceful shutdown initiated (Ctrl+C pressed)")
    print("Waiting for current batch to complete...")
    # The actual shutdown logic is handled in the main processing loop
    signal.signal(signum, signal.default_int_handler)

def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print system information
    sys_info = get_system_info()
    print("\nSystem Information:")
    print(f"GPU: {sys_info['gpu_name']}")
    print(f"GPU Available: {sys_info['gpu_available']}")
    print(f"Initial Memory Usage: {sys_info['memory_usage']:.2f} MB")
    
    # Initialize ChromaDB with specific embedding dimension
    client = chromadb.Client()
    collection = client.create_collection(
        name="log_docs",
        metadata={"hnsw:space": "cosine", "dimension": 1024}  # Match embedding dimension
    )

    # Path to your log file
    log_file_path = "./data/out.log"
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    # Read and chunk the log file
    log_chunks = read_log_file(log_file_path)

    # Configure Ollama based on GPU availability
    ollama_config = {
        "gpu": sys_info['gpu_available'],
        "numa": False,
        "threads": 4,
        "context_window": 4096,
        "batch_size": 8,
    }
    
    # Store each chunk in the vector database with progress bar
    print("\nProcessing log chunks and generating embeddings...")
    print("Configuration:", ollama_config)
    print(f"Using {'GPU' if sys_info['gpu_available'] else 'CPU'} mode")
    start_time = time.time()
    
    # Initialize processing state
    batch_size = 10  # Process multiple chunks at once
    current_batch = []
    error_count = 0
    max_retries = 3
    processed_count = 0
    total_chunks = len(log_chunks)
    
    # Create state file to track progress
    state_file = "log_analyzer_state.txt"
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                processed_count = int(f.read().strip())
                print(f"\nResuming from chunk {processed_count}")
                # Skip already processed chunks
                log_chunks = log_chunks[processed_count:]
    except Exception as e:
        print(f"\nError reading state file: {e}")
    try:
        for i, chunk in tqdm(enumerate(log_chunks), total=len(log_chunks), 
                            desc="Processing chunks", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            current_batch.append(chunk)
            
            # Process batch when it reaches batch_size or at the end
            if len(current_batch) >= batch_size or i == len(log_chunks) - 1:
                if i > 0 and i % 100 == 0:
                    sys_info = get_system_info()
                    print(f"\nProgress Update:")
                    print(f"Memory Usage: {sys_info['memory_usage']:.2f} MB")
                    print(f"CPU Usage: {sys_info['cpu_percent']}%")
                    print(f"Processed {i} chunks in {(time.time() - start_time):.2f} seconds")
                
                for retry in range(max_retries):
                    try:
                        # Process entire batch at once
                        # Process each chunk in the batch individually
                        embeddings = []
                        for chunk in current_batch:
                            try:
                                response = ollama.embeddings(
                                    model="mxbai-embed-large",
                                    prompt=chunk,
                                    options=ollama_config
                                )
                                # Verify embedding dimension
                                if len(response["embedding"]) != 1024:
                                    print(f"\nWarning: Got embedding dimension {len(response['embedding'])}, expected 1024")
                                    embeddings.append([0.0] * 1024)  # Use zero vector as fallback
                                else:
                                    embeddings.append(response["embedding"])
                            except Exception as e:
                                print(f"\nError embedding chunk: {e}")
                                embeddings.append([0.0] * 1024)  # Use zero vector as fallback
                        
                        # Add batch to collection
                        collection.add(
                            ids=[str(i-len(current_batch)+j+1) for j in range(len(current_batch))],
                            embeddings=embeddings,
                            documents=current_batch
                        )
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            print(f"\nError processing batch at chunk {i}: {e}")
                            error_count += len(current_batch)
                            embeddings = [[0.0] * 1024 for _ in current_batch]  # Fallback embeddings
                        time.sleep(1)  # Brief pause before retry
                
                # Update and save progress
                batch_processed = len(current_batch)
                processed_count += batch_processed
                
                try:
                    with open(state_file, 'w') as f:
                        f.write(str(processed_count))
                    if processed_count % 100 == 0:
                        print(f"\nProgress saved: {processed_count}/{total_chunks} chunks processed ({(processed_count/total_chunks)*100:.1f}%)")
                except Exception as e:
                    print(f"\nError saving state: {e}")
                
                current_batch = []  # Clear the batch
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving progress...")
        # Save any remaining chunks in the current batch
        if current_batch:
            try:
                print(f"Saving final batch of {len(current_batch)} chunks...")
                embeddings = [[0.0] * 1024 for _ in current_batch]  # Match 1024 dimension
                collection.add(
                    ids=[str(i-len(current_batch)+j+1) for j in range(len(current_batch))],
                    embeddings=embeddings,
                    documents=current_batch
                )
                processed_count += len(current_batch)
                with open(state_file, 'w') as f:
                    f.write(str(processed_count))
                print("Final batch saved successfully")
            except Exception as e:
                print(f"Error saving final batch: {e}")
        
        print(f"\nFinal Statistics:")
        print(f"- Total chunks processed: {processed_count}/{total_chunks}")
        print(f"- Progress: {(processed_count/total_chunks)*100:.1f}%")
        print(f"- Processing rate: {processed_count/((time.time() - start_time)/60):.1f} chunks/minute")
        print(f"\nProgress saved. You can resume processing by running the script again.")
        print("\nAnalyzing processed chunks before exit...")
        
        # Create a prompt for analysis
        prompt = "Please provide a 10-sentence summary of the key events and information in these log entries."

        try:
            # Get embedding for the prompt
            response = ollama.embeddings(
                prompt=prompt,
                model="mxbai-embed-large"
            )
            
            # Query the collection with what we have so far
            results = collection.query(
                query_embeddings=[response["embedding"]],
                n_results=5
            )

            # Combine the retrieved chunks
            context = "\n".join(results['documents'][0])
            
            # Generate the summary
            print("\nGenerating log analysis...")
            output = ollama.generate(
                model="llama3.2:3b",
                prompt=f"Based on these log entries:\n\n{context}\n\nPlease provide a clear and concise 10-sentence summary of the key events, issues, and information found in these logs. Focus on the most important patterns and notable events.",
                options={
                    "gpu": False,  # Use CPU for final analysis
                    "temperature": 0.7
                }
            )
            
            print("\nLog Analysis Results:")
            print("-" * 50)
            print(output['response'])
            
        except Exception as e:
            print(f"\nError during final analysis: {e}")
        
        return  # Exit after analysis

    # Create a prompt for summarization
    prompt = "Please provide a 10-sentence summary of the key events and information in these log entries."

    # Get embedding for the prompt
    try:
        response = ollama.embeddings(
            prompt=prompt,
            model="mxbai-embed-large"
        )
        
        # Verify embedding dimension
        embedding_dim = len(response["embedding"])
        print(f"\nEmbedding dimension: {embedding_dim}")
        
        # Retrieve the most relevant chunks
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=5  # Increased to get more context for summarization
        )

        # Combine the retrieved chunks
        context = "\n".join(results['documents'][0])
    except Exception as e:
        print(f"Error during embedding or query: {e}")
        return

    # Generate the summary
    print("\nGenerating summary...")
    try:
        output = ollama.generate(
            model="llama3.2:3b",
            prompt=f"Based on these log entries:\n\n{context}\n\nPlease provide a clear and concise 10-sentence summary of the key events, issues, and information found in these logs. Focus on the most important patterns and notable events.",
            options={
                "gpu": False,  # Use CPU since GPU failed
                "temperature": 0.7
            }
        )
    except KeyboardInterrupt:
        print("\nSummary generation interrupted")
        return
    except Exception as e:
        print(f"\nError generating summary: {e}")
        return

    print("\nLog Summary:")
    print("-" * 50)
    print(output['response'])

if __name__ == "__main__":
    main()
