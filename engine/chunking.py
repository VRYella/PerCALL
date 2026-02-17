"""
Chunking Module
===============

Sequence chunking logic for processing large DNA sequences.
Provides parallel chunk processing with overlap management and deduplication.

Extracted from nonbscanner.py for focused chunking logic following modular architecture.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Chunking configuration constants
CHUNK_THRESHOLD = 100000  # 100KB - only chunk very large sequences
DEFAULT_CHUNK_SIZE = 500000  # 500KB chunks for optimal performance
DEFAULT_CHUNK_OVERLAP = 1000  # 1KB overlap (optimal balance)


def calculate_chunks(seq_len: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int]]:
    """
    Calculate chunk boundaries for a sequence.
    
    Args:
        seq_len: Total sequence length
        chunk_size: Size of each chunk in bp
        chunk_overlap: Overlap between chunks in bp
        
    Returns:
        List of (start, end) tuples representing chunk boundaries
    """
    chunks = []
    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        chunks.append((start, end))
        if end >= seq_len:
            break
        # Move to next chunk with overlap
        start = end - chunk_overlap
    
    return chunks


def adjust_motif_positions(motifs: List[Dict[str, Any]], chunk_start: int, 
                          sequence_name: str, chunk_name: str) -> None:
    """
    Adjust motif positions from chunk-relative to sequence-absolute coordinates.
    Modifies motifs in place.
    
    Args:
        motifs: List of motif dictionaries to adjust
        chunk_start: Start position of chunk in full sequence
        sequence_name: Name of the full sequence
        chunk_name: Name of the chunk (to be replaced)
    """
    for motif in motifs:
        motif['Start'] = motif['Start'] + chunk_start
        motif['End'] = motif['End'] + chunk_start
        motif['ID'] = motif['ID'].replace(chunk_name, sequence_name)
        motif['Sequence_Name'] = sequence_name


def process_sequence_chunks(sequence: str, sequence_name: str,
                           scanner: Any,
                           chunk_size: int = DEFAULT_CHUNK_SIZE,
                           chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                           progress_callback: Optional[Callable[[int, int, int, float, float], None]] = None,
                           use_parallel: bool = True) -> List[Dict[str, Any]]:
    """
    Analyze a large sequence by processing it in chunks.
    
    This function divides the sequence into overlapping chunks, analyzes each chunk,
    and then merges the results while removing duplicate motifs from overlap regions.
    
    Args:
        sequence: DNA sequence to analyze
        sequence_name: Identifier for the sequence
        scanner: NonBScanner instance to use for analysis
        chunk_size: Size of each chunk in bp
        chunk_overlap: Overlap between chunks
        progress_callback: Optional callback for progress tracking
                          Signature: (chunk_num, total_chunks, bp_processed, elapsed, throughput)
        use_parallel: Enable parallel chunk processing
        
    Returns:
        List of deduplicated motif dictionaries sorted by position
    """
    seq_len = len(sequence)
    
    # Calculate chunks
    chunks = calculate_chunks(seq_len, chunk_size, chunk_overlap)
    total_chunks = len(chunks)
    all_motifs = []
    start_time = time.time()
    bp_processed = 0
    
    # Process chunks (parallel or sequential)
    if use_parallel and total_chunks > 1:
        # Parallel chunk processing using ThreadPoolExecutor
        # Optimize worker count: use fewer workers for large chunks to reduce memory pressure
        max_workers = min(total_chunks, min(os.cpu_count() or 4, 8))  # Cap at 8 for memory efficiency
        
        def process_chunk(chunk_info):
            chunk_idx, (chunk_start, chunk_end) = chunk_info
            chunk_seq = sequence[chunk_start:chunk_end]
            chunk_name = f"{sequence_name}_chunk{chunk_idx}"
            
            # Analyze chunk
            chunk_motifs = scanner.analyze_sequence(chunk_seq, chunk_name)
            
            # Adjust motif positions to full sequence coordinates
            adjust_motif_positions(chunk_motifs, chunk_start, sequence_name, chunk_name)
            
            # Free chunk sequence memory immediately
            del chunk_seq
            
            return chunk_idx, chunk_end - chunk_start, chunk_motifs
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, (i, chunk)): i 
                      for i, chunk in enumerate(chunks)}
            
            results_by_idx = {}
            for future in as_completed(futures):
                chunk_idx, chunk_len, chunk_motifs = future.result()
                results_by_idx[chunk_idx] = chunk_motifs
                bp_processed += chunk_len
                
                # Call progress callback
                if progress_callback is not None:
                    elapsed = time.time() - start_time
                    throughput = bp_processed / elapsed if elapsed > 0 else 0
                    progress_callback(chunk_idx + 1, total_chunks, bp_processed, elapsed, throughput)
                
                # Trigger garbage collection every 10 chunks to free memory
                if (chunk_idx + 1) % 10 == 0:
                    gc.collect()
            
            # Collect results in order and free intermediate dict
            for i in range(total_chunks):
                all_motifs.extend(results_by_idx.get(i, []))
            
            # Free the results dictionary
            del results_by_idx
            gc.collect()
    else:
        # Sequential chunk processing
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_seq = sequence[chunk_start:chunk_end]
            chunk_name = f"{sequence_name}_chunk{chunk_idx}"
            
            # Analyze chunk
            chunk_motifs = scanner.analyze_sequence(chunk_seq, chunk_name)
            
            # Adjust motif positions to full sequence coordinates
            adjust_motif_positions(chunk_motifs, chunk_start, sequence_name, chunk_name)
            
            all_motifs.extend(chunk_motifs)
            bp_processed += chunk_end - chunk_start
            
            # Free chunk sequence memory immediately
            del chunk_seq
            del chunk_motifs
            
            # Trigger garbage collection every 10 chunks to free memory
            if (chunk_idx + 1) % 10 == 0:
                gc.collect()
            
            # Call progress callback
            if progress_callback is not None:
                elapsed = time.time() - start_time
                throughput = bp_processed / elapsed if elapsed > 0 else 0
                progress_callback(chunk_idx + 1, total_chunks, bp_processed, elapsed, throughput)
    
    # Note: Deduplication is handled by engine.merging module
    # Sort by position
    all_motifs.sort(key=lambda x: x.get('Start', 0))
    
    return all_motifs


def should_chunk_sequence(sequence: str, threshold: int = CHUNK_THRESHOLD) -> bool:
    """
    Determine if a sequence should be chunked based on its length.
    
    Args:
        sequence: DNA sequence string
        threshold: Length threshold in bp
        
    Returns:
        True if sequence should be chunked, False otherwise
    """
    return len(sequence) > threshold
