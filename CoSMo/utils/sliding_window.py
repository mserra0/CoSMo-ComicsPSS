import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from utils.metrics import calculate_mndd, panoptic_quality_metrics

def create_sliding_windows(features, labels, window_size, stride):
    """
    Create sliding windows from a sequence of features and labels.
    
    Args:
        features: Tensor of shape (seq_len, feature_dim)
        labels: Tensor of shape (seq_len,)
        window_size: Size of each window
        stride: Step size between windows
    
    Returns:
        List of tuples (window_features, window_labels, start_idx)
    """
    windows = []
    seq_len = len(features)
    
    if seq_len <= window_size:
        # If sequence is shorter than window, return the whole sequence
        return [(features, labels, 0)]
    
    for start_idx in range(0, seq_len - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_features = features[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]
        windows.append((window_features, window_labels, start_idx))
    
    final_end_idx = start_idx + window_size
    if final_end_idx < seq_len:
        leftover_start = seq_len - window_size
        leftover_features = features[leftover_start:]
        leftover_labels = labels[leftover_start:]
        windows.append((leftover_features, leftover_labels, leftover_start))

    return windows

def aggregate_predictions(predictions_list, sequence_length):
    """
    Aggregate predictions from overlapping windows using majority voting.
    
    Args:
        predictions_list: List of tuples (predictions, start_idx)
        sequence_length: Original sequence length
    
    Returns:
        Aggregated predictions as numpy array
    """
    # Initialize vote counting arrays
    vote_counts = np.zeros((sequence_length, 5))  # Assuming 4 classes
    prediction_counts = np.zeros(sequence_length)
    
    for predictions, start_idx in predictions_list:
        for i, pred in enumerate(predictions):
            page_idx = start_idx + i
            if page_idx < sequence_length:
                vote_counts[page_idx, pred] += 1
                prediction_counts[page_idx] += 1
    
    # Get final predictions by majority vote
    final_predictions = np.argmax(vote_counts, axis=1)
    
    # Handle any positions with no predictions (shouldn't happen with proper stride)
    no_pred_mask = prediction_counts == 0
    if np.any(no_pred_mask):
        print(f"Warning: {np.sum(no_pred_mask)} positions have no predictions")
        # Fill with most common class or handle as needed
        final_predictions[no_pred_mask] = 0  # Default to class 0
    
    return final_predictions

def evaluate_with_sliding_window(model, dataset, device, window_size, stride, batch_size=32):
    """
    Evaluate model using sliding window approach on individual books.
    
    Args:
        model: Trained model
        dataset: Dataset containing books
        device: Device to run evaluation on
        window_size: Size of sliding window
        stride: Stride for sliding window
        batch_size: Batch size for processing windows
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_book_predictions = []
    all_book_labels = []
    book_mndd_scores = []
    docs_prec = []
    docs_recall = []
    docs_f1 = []
    docs_sq = []
    docs_pq = []
    
    print(f"Evaluating {len(dataset)} books with sliding window (size={window_size}, stride={stride})...")
    
    with torch.no_grad():
        for book_idx in tqdm.tqdm(range(len(dataset)), desc="Evaluating books"):
            # Get individual book data
            book_data = dataset[book_idx]
            book_features = book_data['features']  # Shape: (seq_len, feature_dim)
            book_labels = book_data['page_labels']  # Shape: (seq_len,)
            attention_mask = book_data['attention_mask']
            
            book_features = book_features.to(device)
            book_labels = book_labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # Find actual sequence length (non-padded)
            actual_length = attention_mask.sum().item()
            book_features = book_features[:actual_length]
            book_labels = book_labels[:actual_length]
            
            if actual_length == 0:
                continue
            
            # Create sliding windows for this book
            windows = create_sliding_windows(book_features, book_labels, window_size, stride)
            
            # Collect predictions for all windows of this book
            book_window_predictions = []
            
            # Process windows in batches
            window_features_batch = []
            window_labels_batch = []
            window_start_indices = []
            
            for window_idx, (window_features, window_labels, start_idx) in enumerate(windows):
                window_features_batch.append(window_features)
                window_labels_batch.append(window_labels)
                window_start_indices.append(start_idx)
                
                # Process batch when it's full or at the end
                if len(window_features_batch) == batch_size or window_idx == len(windows) - 1:
                    # Stack windows into batch
                    batch_features = torch.stack(window_features_batch).to(device)
                    batch_attention_mask = torch.ones(batch_features.shape[:2]).to(device)
                    
                    # Get predictions
                    logits = model(batch_features, batch_attention_mask)
                    predictions = logits.argmax(dim=2)
                    
                    # Store predictions with their start indices
                    for i, start_idx in enumerate(window_start_indices):
                        window_preds = predictions[i].cpu().numpy()
                        book_window_predictions.append((window_preds, start_idx))
                    
                    # Reset batch
                    window_features_batch = []
                    window_labels_batch = []
                    window_start_indices = []
            
            # Aggregate predictions for this book
            if book_window_predictions:
                aggregated_preds = aggregate_predictions(book_window_predictions, actual_length)
                true_labels = book_labels.cpu().numpy()
                
                # Calculate metrics for this book
                book_score = calculate_mndd(aggregated_preds, true_labels)
                book_mndd_scores.append(book_score)
                
                metrics = panoptic_quality_metrics(aggregated_preds, true_labels)
                docs_prec.append(metrics["precision"])
                docs_recall.append(metrics["recall"])
                docs_f1.append(metrics["f1"])
                docs_sq.append(metrics["sq"])
                docs_pq.append(metrics["pq"])
                
                # Store for overall metrics
                all_book_predictions.extend(aggregated_preds)
                all_book_labels.extend(true_labels)
    
    return {
        'book_mndd_scores': book_mndd_scores,
        'docs_prec': docs_prec,
        'docs_recall': docs_recall,
        'docs_f1': docs_f1,
        'docs_sq': docs_sq,
        'docs_pq': docs_pq,
        'all_predictions': all_book_predictions,
        'all_labels': all_book_labels
    }
    




def create_multimodal_sliding_windows(textual_features, visual_features, labels, window_size, stride):
    """
    Create sliding windows from sequences of textual and visual features and labels.
    
    Args:
        textual_features: Tensor of shape (seq_len, text_feature_dim)
        visual_features: Tensor of shape (seq_len, visual_feature_dim)
        labels: Tensor of shape (seq_len,)
        window_size: Size of each window
        stride: Step size between windows
    
    Returns:
        List of tuples (window_textual_features, window_visual_features, window_labels, start_idx)
    """
    windows = []
    seq_len = len(visual_features)
    
    if seq_len <= window_size:
        # If sequence is shorter than window, return the whole sequence
        return [(textual_features, visual_features, labels, 0)]
    
    for start_idx in range(0, seq_len - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_textual = textual_features[start_idx:end_idx]
        window_visual = visual_features[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]
        windows.append((window_textual, window_visual, window_labels, start_idx))
    
    # Handle the final segment if there's a leftover portion
    final_end_idx = start_idx + window_size
    if final_end_idx < seq_len:
        leftover_start = seq_len - window_size
        leftover_textual = textual_features[leftover_start:]
        leftover_visual = visual_features[leftover_start:]
        leftover_labels = labels[leftover_start:]
        windows.append((leftover_textual, leftover_visual, leftover_labels, leftover_start))

    return windows

def aggregate_predictions(predictions_list, sequence_length):
    """
    Aggregate predictions from overlapping windows using majority voting.
    
    Args:
        predictions_list: List of tuples (predictions, start_idx)
        sequence_length: Original sequence length
    
    Returns:
        Aggregated predictions as numpy array
    """
    # Initialize vote counting arrays
    vote_counts = np.zeros((sequence_length, 5))  # Assuming 4 classes (0-4)
    prediction_counts = np.zeros(sequence_length)
    
    for predictions, start_idx in predictions_list:
        for i, pred in enumerate(predictions):
            page_idx = start_idx + i
            if page_idx < sequence_length:
                vote_counts[page_idx, pred] += 1
                prediction_counts[page_idx] += 1
    
    # Get final predictions by majority vote
    final_predictions = np.argmax(vote_counts, axis=1)
    
    # Handle any positions with no predictions (shouldn't happen with proper stride)
    no_pred_mask = prediction_counts == 0
    if np.any(no_pred_mask):
        print(f"Warning: {np.sum(no_pred_mask)} positions have no predictions")
        # Fill with most common class or handle as needed
        final_predictions[no_pred_mask] = 0  # Default to class 0
    
    return final_predictions

def evaluate_multimodal_with_sliding_window(model, dataset, device, window_size, stride, batch_size=32):
    """
    Evaluate multimodal model using sliding window approach on individual books.
    
    Args:
        model: Trained multimodal model
        dataset: Multimodal dataset containing books
        device: Device to run evaluation on
        window_size: Size of sliding window
        stride: Stride for sliding window
        batch_size: Batch size for processing windows
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_book_predictions = []
    all_book_labels = []
    book_mndd_scores = []
    docs_prec = []
    docs_recall = []
    docs_f1 = []
    docs_sq = []
    docs_pq = []
    
    print(f"Evaluating {len(dataset)} books with multimodal sliding window (size={window_size}, stride={stride})...")
    
    with torch.no_grad():
        for book_idx in tqdm.tqdm(range(len(dataset)), desc="Evaluating books"):
            book_data = dataset[book_idx]
            textual_features = book_data['textual_features']  # Shape: (seq_len, text_feature_dim)
            visual_features = book_data['visual_features']    # Shape: (seq_len, visual_feature_dim)
            book_labels = book_data['page_labels']            # Shape: (seq_len,)
            attention_mask = book_data['attention_mask']
            
            textual_features = textual_features.to(device)
            visual_features = visual_features.to(device)
            book_labels = book_labels.to(device)
            attention_mask = attention_mask.to(device)
            
            actual_length = attention_mask.sum().item()
            textual_features = textual_features[:actual_length]
            visual_features = visual_features[:actual_length]
            book_labels = book_labels[:actual_length]
            
            if actual_length == 0:
                continue
            
            windows = create_multimodal_sliding_windows(
                textual_features, visual_features, book_labels, window_size, stride
            )
            
            book_window_predictions = []
            
            window_textual_batch = []
            window_visual_batch = []
            window_labels_batch = []
            window_start_indices = []
            
            for window_idx, (window_textual, window_visual, window_labels, start_idx) in enumerate(windows):
                window_textual_batch.append(window_textual)
                window_visual_batch.append(window_visual)
                window_labels_batch.append(window_labels)
                window_start_indices.append(start_idx)
                
                if len(window_textual_batch) == batch_size or window_idx == len(windows) - 1:
                    batch_textual = torch.stack(window_textual_batch).to(device)
                    batch_visual = torch.stack(window_visual_batch).to(device)
                    batch_attention_mask = torch.ones(batch_textual.shape[:2]).to(device)
         
                    logits = model(batch_textual, batch_visual, batch_attention_mask)
                    predictions = logits.argmax(dim=2)
                    
                    for i, start_idx in enumerate(window_start_indices):
                        window_preds = predictions[i].cpu().numpy()
                        book_window_predictions.append((window_preds, start_idx))

                    window_textual_batch = []
                    window_visual_batch = []
                    window_labels_batch = []
                    window_start_indices = []
            
            if book_window_predictions:
                aggregated_preds = aggregate_predictions(book_window_predictions, actual_length)
                true_labels = book_labels.cpu().numpy()
                
                book_score = calculate_mndd(aggregated_preds, true_labels)
                book_mndd_scores.append(book_score)
                
                metrics = panoptic_quality_metrics(aggregated_preds, true_labels)
                docs_prec.append(metrics["precision"])
                docs_recall.append(metrics["recall"])
                docs_f1.append(metrics["f1"])
                docs_sq.append(metrics["sq"])
                docs_pq.append(metrics["pq"])
                

                all_book_predictions.extend(aggregated_preds)
                all_book_labels.extend(true_labels)
    
    return {
        'book_mndd_scores': book_mndd_scores,
        'docs_prec': docs_prec,
        'docs_recall': docs_recall,
        'docs_f1': docs_f1,
        'docs_sq': docs_sq,
        'docs_pq': docs_pq,
        'all_predictions': all_book_predictions,
        'all_labels': all_book_labels
    }