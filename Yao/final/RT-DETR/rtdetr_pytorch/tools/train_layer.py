
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import DetSolver
from tools.weight_loader import load_rtdetr_weights
from src.data.coco.coco_layer_dataset import CocoLayerOrderDataset  # Register custom dataset

def save_layer_prediction_examples(model, dataloader, device, epoch, save_dir, num_examples=4):
    """
    Adapted from ml-depth-pro for RT-DETR output format.
    RT-DETR outputs a dict 'pred_layers' [B, N, 1].
    We need to visualize this.
    Note: RT-DETR predicts per-object layer values for 300 queries, NOT a dense depth map.
    The user's previous code visualized dense depth maps because it was a pixel-wise model (DepthPro).
    RT-DETR is an object detector. 
    VISUALIZATION STRATEGY:
    - We cannot easily show a "Depth Map" image from 300 boxes unless we rasterize them?
    - OR, do we just show the bbox and the predicted layer value?
    - User request: "keep consistency... save examples".
    - But data is different now. The input has dense depth channel.
    - Maybe visualize the INPUT depth channel (Channel 3)?
    - And for prediction, maybe overlay boxes with color coding for depth?
    
    Let's visualize:
    1. Input RGB
    2. Input Depth Channel (GT dense map)
    3. Prediction: Bounding boxes with colors mapping to predicted layer value.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = save_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random indices
    # We can't easily index dataloader, so just take the first batch (shuffle is usually on) 
    # or iterate to find one.
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 1: break # Just one batch is enough for 4 examples usually
            
            # batch is (images, targets)
            # images: tensor [B, 4, H, W]
            # targets: list of dicts
            
            images, targets = batch 
            images = images.to(device)
            # targets moved to device inside model usually, but we need them here?
            # actually DetSolver moves them.
            
            # Forward
            outputs = model(images)
            # outputs: {'pred_logits': [B, 300, C], 'pred_boxes': [B, 300, 4], 'pred_layers': [B, 300, 1]}
            
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            pred_layers = outputs['pred_layers']
            
            # For each image in batch (up to num_examples)
            B = images.shape[0]
            for b in range(min(B, num_examples)):
                # 1. Prepare Images
                img_tensor = images[b].cpu() # (4, H, W)
                rgb = img_tensor[:3].permute(1, 2, 0).numpy() # (H, W, 3)
                depth_in = img_tensor[3].numpy() # (H, W)
                
                # Denormalize RGB? COCO usually mean/std normalized. 
                # If transforms used 'Normalize', we should reverse it.
                # Assuming standard ImageNet mean/std used in config?
                # Actually, earlier we checked transforms.yml, it was empty '~'.
                # But 'coco_detection.yml' had 'transforms' section.
                # If no normalize is defined in config, it's [0,1].
                # User's ml-depth-pro normalized to 0.5 mean/std.
                # RT-DETR usually just ToTensor (0-1).
                # Let's assume 0-1 for safety or clip.
                rgb = np.clip(rgb, 0, 1)
                
                # 2. GT Info
                # target = targets[b] # dict with 'layers', 'boxes'
                # Accessing raw target might be tricky if it's already on GPU or transformed?
                # Dataloader yields CPU tensors usually.
                
                gt_layers = targets[b]['layers'] # Tensor
                gt_boxes = targets[b]['boxes'] # Tensor
                # Convert boxes to xyxy if cxcywh?
                # RT-DETR targets are usually normalized cxcywh.
                # We need to convert to absolute xyxy for plotting.
                H, W = rgb.shape[:2]
                gt_boxes_abs = gt_boxes.clone()
                # cxcywh -> xyxy
                cx, cy, w, h = gt_boxes_abs.unbind(-1)
                b_x1 = (cx - 0.5 * w) * W
                b_y1 = (cy - 0.5 * h) * H
                b_x2 = (cx + 0.5 * w) * W
                b_y2 = (cy + 0.5 * h) * H
                
                # 3. Pred Info
                # Filter low confidence
                prob = pred_logits[b].sigmoid()
                scores, labels = prob.max(-1)
                keep = scores > 0.5
                
                p_boxes = pred_boxes[b][keep].cpu()
                p_layers = pred_layers[b][keep].cpu().flatten()
                
                p_boxes_abs = p_boxes.clone()
                cx, cy, w, h = p_boxes_abs.unbind(-1)
                pb_x1 = (cx - 0.5 * w) * W
                pb_y1 = (cy - 0.5 * h) * H
                pb_x2 = (cx + 0.5 * w) * W
                pb_y2 = (cy + 0.5 * h) * H
                
                # PLOTTING
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Input RGB
                axes[0].imshow(rgb)
                axes[0].set_title("Input RGB")
                axes[0].axis('off')
                
                # Input Depth (GT Channel)
                im = axes[1].imshow(depth_in, cmap='viridis', vmin=0, vmax=1)
                axes[1].set_title("Input Depth Map (Ch 4)")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                # Predictions (Visualized as Boxes colored by Layer)
                axes[2].imshow(rgb)
                # Draw Predicted Boxes
                if len(p_boxes) > 0:
                    import matplotlib.patches as patches
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap('viridis')
                    
                    for k in range(len(p_boxes)):
                        x1, y1 = pb_x1[k], pb_y1[k]
                        w_box = pb_x2[k] - x1
                        h_box = pb_y2[k] - y1
                        
                        l_val = p_layers[k].item()
                        # normalize l_val for color lookup (assuming roughly 0-1 or -1~1, but user normalized to median/std?)
                        # User said: "GT layer index d ... normalized d_hat = (d-m)/s"
                        # So it's not 0-1. It's standardized.
                        # We just clamp to e.g. -3 to 3 for viz?
                        norm_l = (l_val + 2) / 4 # map -2..2 to 0..1 roughly
                        color = cmap(np.clip(norm_l, 0, 1))
                        
                        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor='none')
                        axes[2].add_patch(rect)
                        axes[2].text(x1, y1, f"{l_val:.2f}", color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))
                
                axes[2].set_title("Pred Boxes (Color=Layer)")
                axes[2].axis('off')
                
                plt.suptitle(f"Epoch {epoch} - Ex {i*num_examples + b}", fontsize=16)
                plt.savefig(examples_dir / f"epoch_{epoch:04d}_ex_{i*num_examples + b}.png")
                plt.close()

class LayerDetSolver(DetSolver):
    def fit(self):
        print("Start training")
        self.train()  # This initializes dataloaders
        args = self.cfg 
        
        import time
        import json
        import datetime
        from src.solver.det_engine import train_one_epoch, evaluate
        from src.misc import dist

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # Save initial checkpoint (before any training)
        print("Saving initial checkpoint (epoch -1, untrained model)...")
        if self.output_dir:
            initial_checkpoint_path = self.output_dir / 'checkpoint_initial.pth'
            dist.save_on_master(self.state_dict(-1), initial_checkpoint_path)
            print(f"Initial checkpoint saved to {initial_checkpoint_path}")

        from src.data import get_coco_api_from_dataset
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # TRAIN
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            # Visualization disabled to prevent training interruption

            # SAVE CHECKPOINTS
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            # EVAL - wrapped in try-except to handle position embedding size mismatches
            try:
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
                )
                
                # LOGGING
                for k in test_stats.keys():
                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat[k] = test_stats[k][0]
                        best_stat['epoch'] = epoch
                        
                # SAVE LOG
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
                print('best_stat: ', best_stat) # Moved here to be part of successful eval logging
            except RuntimeError as e:
                if "size of tensor" in str(e) and "must match" in str(e):
                    print(f"\n⚠️  Skipping evaluation for epoch {epoch} due to position embedding error")
                    print(f"   Error: {str(e)[:100]}...")
                    print(f"   Training will continue. This is likely due to variable image sizes.")
                    
                    # Log without test stats
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters,
                                'eval_skipped': True}
                else:
                    # Re-raise if it's a different error
                    raise

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

def main(args):
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    # Use custom solver
    # TASKS map doesn't know about our subclass, so we instantiate manually.
    # But cfg.yaml_cfg calls for 'task: detection' which maps to DetSolver in TASKS.
    # We bypass TASKS lookup and instantiate directly.
    solver = LayerDetSolver(cfg)
    
    # Custom Weight Loading
    if args.pretrained_path:
        # Load 4-channel weights (modifies solver.cfg.model in-place)
        load_rtdetr_weights(solver.cfg.model, args.pretrained_path)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    parser.add_argument('--pretrained-path', type=str, help='Path to 3-channel COCO weights for surgical loading')
    args = parser.parse_args()

    main(args)
