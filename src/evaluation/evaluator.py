import numpy as np
import jieba
import logging
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

class ChineseEvaluator:
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1 if METRICS_AVAILABLE else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) if METRICS_AVAILABLE else None

    def tokenize_chinese(self, text):
        if not text:
            return []
        return list(jieba.cut(text.strip()))

    def compute_bleu(self, predictions, references, n_gram=4):
        if not METRICS_AVAILABLE:
            return 0.0
            
        scores = []
        weights_dict = {
            1: (1.0, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (1/3, 1/3, 1/3, 0),
            4: (0.25, 0.25, 0.25, 0.25)
        }
        
        weights = weights_dict.get(n_gram, weights_dict[4])
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenize_chinese(pred)
            ref_tokens = [self.tokenize_chinese(ref)]
            
            if len(pred_tokens) == 0:
                scores.append(0.0)
                continue
                
            try:
                score = sentence_bleu(
                    ref_tokens, 
                    pred_tokens, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                scores.append(score)
            except:
                scores.append(0.0)
        
        return np.mean(scores) * 100 if scores else 0.0

    def compute_rouge(self, predictions, references):
        if not METRICS_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            pred_text = ' '.join(self.tokenize_chinese(pred))
            ref_text = ' '.join(self.tokenize_chinese(ref))
            
            try:
                scores = self.rouge_scorer.score(ref_text, pred_text)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            except:
                rouge_scores['rouge1'].append(0.0)
                rouge_scores['rouge2'].append(0.0)
                rouge_scores['rougeL'].append(0.0)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']) * 100,
            'rouge2': np.mean(rouge_scores['rouge2']) * 100,
            'rougeL': np.mean(rouge_scores['rougeL']) * 100
        }

    def compute_exact_match(self, predictions, references):
        if not predictions or not references:
            return 0.0
        
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip() == ref.strip())
        return (matches / len(predictions)) * 100

    def compute_token_overlap(self, predictions, references):
        if not predictions or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precisions = []
        recalls = []
        f1s = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self.tokenize_chinese(pred))
            ref_tokens = set(self.tokenize_chinese(ref))
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                precisions.append(1.0)
                recalls.append(1.0)
                f1s.append(1.0)
            elif len(pred_tokens) == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
            else:
                overlap = len(pred_tokens & ref_tokens)
                precision = overlap / len(pred_tokens)
                recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
        
        return {
            'precision': np.mean(precisions) * 100,
            'recall': np.mean(recalls) * 100,
            'f1': np.mean(f1s) * 100
        }

    def compute_diversity_metrics(self, predictions):
        """Compute diversity metrics for predictions."""
        if not predictions:
            return {
                'unique_predictions': 0,
                'prediction_diversity': 0.0,
                'avg_prediction_length': 0.0
            }
        
        unique_predictions = len(set(predictions))
        pred_lengths = [len(self.tokenize_chinese(p)) for p in predictions]
        
        return {
            'unique_predictions': unique_predictions,
            'prediction_diversity': unique_predictions / len(predictions),
            'avg_prediction_length': np.mean(pred_lengths) if pred_lengths else 0.0
        }

    def decode_batch(self, generated_ids: torch.Tensor, labels: torch.Tensor, tokenizer) -> Tuple[List[str], List[str]]:
        """Decode generated IDs and labels to text."""
        predictions = []
        targets = []
        
        for i in range(len(generated_ids)):
            pred_text = tokenizer.decode(
                generated_ids[i], 
                skip_special_tokens=True
            ).strip()
            predictions.append(pred_text)
            
            target_ids = labels[i][labels[i] != -100]
            target_text = tokenizer.decode(
                target_ids, 
                skip_special_tokens=True
            ).strip()
            targets.append(target_text)
        
        return predictions, targets

    def create_example_list(self, predictions: List[str], targets: List[str]) -> List[Dict]:
        """Create detailed examples with comparison metrics."""
        examples = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            examples.append({
                'index': i,
                'prediction': pred,
                'target': target,
                'exact_match': pred.strip() == target.strip(),
                'pred_length': len(pred.split()),
                'target_length': len(target.split())
            })
        return examples

    def evaluate_model(self, model, val_loader, tokenizer, config: Dict, device: str) -> Dict:
        model.eval()
        
        all_predictions = []
        all_targets = []
        total_val_loss = 0.0
        total_samples = 0
        
        # Use simpler generation config for evaluation
        gen_config = {
            'max_new_tokens': 20,
            'min_new_tokens': 3,
            'do_sample': True,
            'temperature': 0.8,  # Lower temperature for more focused generation
            'top_p': 0.9,
            'top_k': 30,
            'repetition_penalty': 1.2,
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                try:
                    # Simple forward pass to get loss
                    eeg_data = [region.to(device) for region in batch['eeg']]
                    labels = batch['labels'].to(device)
                    text_ids = batch.get('decoder_input_ids', labels[:, :-1]).to(device)
                    
                    outputs = model(eeg_data=eeg_data, input_ids=text_ids, labels=labels)
                    loss = outputs.get('loss', torch.tensor(0.0))
                    
                    batch_size = labels.size(0)
                    total_val_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    # Generation with error handling
                    try:
                        generated_ids = model.generate(eeg_data=eeg_data, **gen_config)
                        
                        # Decode predictions and targets
                        for i in range(batch_size):
                            # Decode prediction
                            pred_ids = generated_ids[i].cpu().numpy()
                            pred_ids = pred_ids[pred_ids >= 0]
                            pred_ids = pred_ids[pred_ids < tokenizer.vocab_size]
                            
                            if len(pred_ids) > 0:
                                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                            else:
                                pred_text = ""
                            
                            # Decode target
                            target_ids = labels[i].cpu().numpy()
                            target_ids = target_ids[target_ids >= 0]
                            target_ids = target_ids[target_ids < tokenizer.vocab_size]
                            
                            if len(target_ids) > 0:
                                target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
                            else:
                                target_text = ""
                            
                            # Only add non-empty pairs
                            if pred_text and target_text:
                                all_predictions.append(pred_text)
                                all_targets.append(target_text)
                                
                    except Exception as e:
                        logger.warning(f"Generation failed in batch: {e}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue
        
        # Compute metrics only if we have valid pairs
        metrics = {}
        metrics['val_loss'] = total_val_loss / max(total_samples, 1)
        
        if all_predictions and all_targets:
            try:
                text_metrics = self.compute_all_metrics(all_predictions, all_targets)
                metrics.update(text_metrics)
            except Exception as e:
                logger.error(f"Error computing text metrics: {e}")
                metrics.update(self._empty_metrics())
        else:
            logger.warning("No valid prediction-target pairs found for evaluation")
            metrics.update(self._empty_metrics())
        
        return metrics

    def _forward_pass(self, model, batch, device) -> Dict:
        """Perform forward pass and return outputs with loss components."""
        try:
            eeg_data = [region.to(device) for region in batch['eeg']]
            targets = batch['labels'].to(device)
            decoder_input_ids = batch.get('decoder_input_ids')
            
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(device)
            
            outputs = model(
                eeg_data=eeg_data,
                decoder_input_ids=decoder_input_ids,
                labels=targets
            )
            
            batch_size = targets.size(0)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                total_loss = outputs.get('loss', torch.tensor(0.0))
                loss_components = {k: v for k, v in outputs.items() if 'loss' in k.lower() and k != 'loss'}
            else:
                total_loss = outputs
                loss_components = {}
            
            return {
                'total_loss': total_loss,
                'loss_components': loss_components,
                'batch_size': batch_size
            }
            
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            return None

    def _compute_comprehensive_metrics(
        self, 
        predictions: List[str], 
        targets: List[str], 
        total_loss: float, 
        total_samples: int,
        component_sums: Dict,
        component_counts: int
    ) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {}
        
        # Loss metrics
        metrics['val_loss'] = total_loss / max(total_samples, 1)
        
        if component_counts > 0:
            for component, sum_value in component_sums.items():
                metrics[f'val_{component}'] = sum_value / component_counts
        
        if predictions and targets:
            try:
                # Text quality metrics
                text_metrics = self.compute_all_metrics(predictions, targets)
                metrics.update(text_metrics)
                
                # Exact match metrics
                exact_matches = sum(1 for p, t in zip(predictions, targets) 
                                  if p.strip() == t.strip())
                metrics.update({
                    'exact_matches': exact_matches,
                    'exact_match_rate': exact_matches / len(predictions),
                    'total_samples': len(predictions)
                })
                
                # Diversity metrics
                diversity_metrics = self.compute_diversity_metrics(predictions)
                metrics.update(diversity_metrics)
                
                # Target diversity for comparison
                target_diversity = self.compute_diversity_metrics(targets)
                metrics['unique_targets'] = target_diversity['unique_predictions']  # reuse function
                
                # Example list for detailed analysis
                metrics['examples'] = self.create_example_list(predictions, targets)
                
            except Exception as e:
                logger.error(f"Error computing text metrics: {e}")
                metrics.update({
                    'bleu_4': 0.0, 'exact_match_rate': 0.0, 'prediction_diversity': 0.0
                })
        else:
            logger.warning("No valid predictions or targets found")
            metrics.update({
                'bleu_4': 0.0, 'exact_match_rate': 0.0, 'prediction_diversity': 0.0
            })
        
        return metrics

    def compute_all_metrics(self, predictions, references):
        """Compute all text quality metrics."""
        if not predictions or not references or len(predictions) != len(references):
            return self._empty_metrics()
        
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) 
                         if p.strip() and r.strip()]
        
        if not filtered_pairs:
            return self._empty_metrics()
        
        filtered_predictions, filtered_references = zip(*filtered_pairs)
        
        metrics = {}
        
        # BLEU scores
        for n in [1, 2, 3, 4]:
            metrics[f'bleu_{n}'] = self.compute_bleu(
                filtered_predictions, filtered_references, n_gram=n
            )
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(filtered_predictions, filtered_references)
        metrics.update({
            'rouge_1_f': rouge_scores['rouge1'],
            'rouge_2_f': rouge_scores['rouge2'],
            'rouge_l_f': rouge_scores['rougeL']
        })
        
        # Token overlap scores
        overlap_scores = self.compute_token_overlap(filtered_predictions, filtered_references)
        metrics.update({
            'token_precision': overlap_scores['precision'],
            'token_recall': overlap_scores['recall'],
            'token_f1': overlap_scores['f1']
        })
        
        # Exact match
        metrics['exact_match'] = self.compute_exact_match(filtered_predictions, filtered_references)
        
        # Length statistics
        pred_lengths = [len(self.tokenize_chinese(p)) for p in filtered_predictions]
        ref_lengths = [len(self.tokenize_chinese(r)) for r in filtered_references]
        
        metrics.update({
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0.0,
            'valid_pairs': len(filtered_pairs),
            'total_pairs': len(predictions)
        })
        
        return metrics

    def _empty_metrics(self):
        """Return empty metrics dict when evaluation fails."""
        return {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0,
            'token_precision': 0.0, 'token_recall': 0.0, 'token_f1': 0.0,
            'exact_match': 0.0, 'avg_pred_length': 0.0, 'avg_ref_length': 0.0,
            'length_ratio': 0.0, 'valid_pairs': 0, 'total_pairs': 0
        }