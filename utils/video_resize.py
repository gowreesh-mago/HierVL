# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import sys
import subprocess
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from tqdm import tqdm

folder_path = '/home/dkoelma1/VisualSearch/Ego4D/v1/full_scale'
output_path = '/hddstore/gmago/Hier_VLM/Ego4D/resized_videos'

os.makedirs(output_path, exist_ok=True)

def videos_resize(videoinfos):
    """Process a single video file with resizing"""
    videoid, videoname = videoinfos
    outname = os.path.join(output_path, videoname)
    
    if os.path.exists(outname):
        return {
            'status': 'skipped',
            'filename': videoname,
            'message': f'{videoname} already exists (skipped)'
        }
    
    inname = os.path.join(folder_path, videoname)
    
    # Optimized ffmpeg command with faster encoding settings
    cmd = [
        "ffmpeg", "-y", "-i", inname,
        "-filter:v", "scale=trunc(oh*a/2)*2:256",
        "-c:a", "copy",
        "-preset", "ultrafast",  # Faster encoding
        "-threads", "1",  # Let each process use 1 thread
        outname
    ]
    
    try:
        # Use subprocess.run for better performance and error handling
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                'status': 'success',
                'filename': videoname,
                'message': f'Successfully resized {videoname}'
            }
        else:
            return {
                'status': 'error',
                'filename': videoname,
                'message': f'FFmpeg error: {result.stderr[:200]}...'  # Truncate long errors
            }
    except Exception as e:
        return {
            'status': 'exception',
            'filename': videoname,
            'message': f'Exception: {str(e)}'
        }

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Build file list more efficiently
    print("Scanning for MP4 files...")
    mp4_list = [item for item in os.listdir(folder_path) if item.endswith('.mp4')]
    file_list = [[id, video] for id, video in enumerate(mp4_list)]
    total_files = len(file_list)
    
    if total_files == 0:
        print("No MP4 files found in the input directory!")
        sys.exit(1)
    
    print(f"Found {total_files} MP4 files to process")
    
    # Use optimal number of processes (typically CPU cores - 1 to leave room for system)
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel processing")
    
    start_time = time.time()
    
    # Initialize counters for results
    results = {
        'success': [],
        'skipped': [],
        'error': [],
        'exception': []
    }
    
    # Create progress bar with custom format
    with tqdm(total=total_files, 
              desc="Processing videos", 
              unit="video",
              ncols=100,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Use ProcessPoolExecutor for better resource management
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks
            future_to_video = {executor.submit(videos_resize, file): file for file in file_list}
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_video):
                try:
                    result = future.result()
                    status = result['status']
                    filename = result['filename']
                    
                    # Categorize results
                    results[status].append(result)
                    
                    # Update progress bar with status-specific description
                    if status == 'success':
                        pbar.set_postfix_str(f"✓ {filename[:30]}...")
                    elif status == 'skipped':
                        pbar.set_postfix_str(f"⏭ {filename[:30]}...")
                    else:  # error or exception
                        pbar.set_postfix_str(f"✗ {filename[:30]}...")
                    
                    # Update progress bar
                    pbar.update(1)
                    
                except Exception as e:
                    # Handle any unexpected errors from futures
                    video_info = future_to_video[future]
                    results['exception'].append({
                        'status': 'exception',
                        'filename': video_info[1],
                        'message': f'Future exception: {str(e)}'
                    })
                    pbar.set_postfix_str(f"✗ Future error")
                    pbar.update(1)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate summary statistics
    success_count = len(results['success'])
    skipped_count = len(results['skipped'])
    error_count = len(results['error'])
    exception_count = len(results['exception'])
    total_processed = success_count + skipped_count + error_count + exception_count
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f"Average time per file: {processing_time/total_files:.2f} seconds")
    print(f"Processing rate: {total_files/processing_time:.1f} files/second")
    print(f"\nResults Summary:")
    print(f"  ✓ Successfully processed: {success_count}")
    print(f"  ⏭ Skipped (already exist): {skipped_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"  ✗ Exceptions: {exception_count}")
    print(f"  📊 Total: {total_processed}/{total_files}")
    
    # Show detailed error information if any
    if error_count > 0 or exception_count > 0:
        print(f"\n{'='*60}")
        print("ERROR DETAILS:")
        print(f"{'='*60}")
        
        # Show first few errors for debugging
        all_errors = results['error'] + results['exception']
        for i, error in enumerate(all_errors[:5]):  # Show first 5 errors
            print(f"{i+1}. {error['filename']}: {error['message']}")
        
        if len(all_errors) > 5:
            print(f"... and {len(all_errors) - 5} more errors")
    
    print(f"{'='*60}")