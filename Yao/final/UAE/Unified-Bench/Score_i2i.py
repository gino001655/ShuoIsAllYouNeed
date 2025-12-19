import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from DINO_v3 import DinoV3Score
from CLIP import Score_Cal as CLIPScore
from DINO_v2 import Score_Cal as DINOv2Score
from LongCLIP import Score_Cal as LongCLIPScore


class ImageSimilarityCalculator:
    """
    Image similarity calculator supporting multiple models
    """
    
    def __init__(self, model_types: List[str] = None, **model_kwargs):
        """
        Initialize similarity calculator
        
        Args:
            model_types: List of model types, supports ["clip", "dinov2", "dinov3", "longclip"]
            **model_kwargs: Model initialization parameters
        """
        if model_types is None:
            model_types = ["clip", "dinov2", "dinov3", "longclip"]
        
        self.model_types = model_types
        self.models = {}
        self._load_models(**model_kwargs)
        
    def _load_models(self, **kwargs) -> None:
        """
        Load all specified types of models
        
        Args:
            **kwargs: Model parameters
        """
        for model_type in self.model_types:
            try:
                print(f"Loading {model_type} model...")
                if model_type == "clip":
                    self.models[model_type] = CLIPScore()
                elif model_type == "dinov2":
                    self.models[model_type] = DINOv2Score()
                elif model_type == "dinov3":
                    self.models[model_type] = DinoV3Score(**kwargs)
                elif model_type == "longclip":
                    self.models[model_type] = LongCLIPScore()
                else:
                    print(f"Warning: Unsupported model type: {model_type}")
                    continue
                print(f"Successfully loaded {model_type} model")
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
                # Continue loading other models, don't stop due to single model failure
    
    def calculate_folder_similarity(self, image_path: str, ref_image_path: str, 
                                  output_file: str = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate similarity between corresponding images in two folders
        
        Args:
            image_path: First image folder path
            ref_image_path: Reference image folder path
            output_file: Result save file path (optional)
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping model names to image similarity mappings
        """
        # Check if folders exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image folder not found: {image_path}")
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Reference image folder not found: {ref_image_path}")
        
        # Get image file lists
        image_files = self._get_image_files(image_path)
        ref_image_files = self._get_image_files(ref_image_path)
        
        print(f"Found {len(image_files)} images in {image_path}")
        print(f"Found {len(ref_image_files)} images in {ref_image_path}")
        
        # Calculate similarity for each model
        all_similarities = {}
        
        for model_name, model in self.models.items():
            print(f"\n=== Processing with {model_name.upper()} model ===")
            similarities = {}
            total_pairs = 0
            
            for img_file in image_files:
                img_name = os.path.basename(img_file)
                # Find corresponding image in reference folder
                ref_file = self._find_corresponding_image(img_name, ref_image_files)
                
                if ref_file:
                    try:
                        similarity = model.calculate_similarity(img_file, ref_file)
                        similarities[img_name] = float(similarity)
                        total_pairs += 1
                        print(f"Processed: {img_name} -> Similarity: {similarity:.4f}")
                    except Exception as e:
                        print(f"Error processing {img_name} with {model_name}: {e}")
                        similarities[img_name] = None
                else:
                    print(f"No corresponding image found for: {img_name}")
                    similarities[img_name] = None
            
            print(f"Processed {total_pairs} image pairs with {model_name}")
            
            # Calculate statistics for this model
            valid_scores = [score for score in similarities.values() if score is not None]
            if valid_scores:
                avg_similarity = sum(valid_scores) / len(valid_scores)
                print(f"{model_name.upper()} - Average similarity: {avg_similarity:.4f}")
                print(f"{model_name.upper()} - Min similarity: {min(valid_scores):.4f}")
                print(f"{model_name.upper()} - Max similarity: {max(valid_scores):.4f}")
            
            all_similarities[model_name] = similarities
        
        # Save results
        if output_file:
            self._save_results(all_similarities, output_file)
            print(f"Results saved to: {output_file}")
        
        return all_similarities
    
    def _get_image_files(self, folder_path: str) -> List[str]:
        """
        Get all image files in the folder
        
        Args:
            folder_path: Folder path
            
        Returns:
            List[str]: List of image file paths
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for file_path in Path(folder_path).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def _find_corresponding_image(self, img_name: str, ref_files: List[str]) -> str:
        """
        Find corresponding image in reference file list
        
        Args:
            img_name: Target image name
            ref_files: Reference file list
            
        Returns:
            str: Corresponding reference image path, returns None if not found
        """
        # Exact match
        for ref_file in ref_files:
            if os.path.basename(ref_file) == img_name:
                return ref_file
        
        # If exact match fails, try matching without extension
        img_name_no_ext = os.path.splitext(img_name)[0]
        for ref_file in ref_files:
            ref_name_no_ext = os.path.splitext(os.path.basename(ref_file))[0]
            if ref_name_no_ext == img_name_no_ext:
                return ref_file
            
            # Try adding prefix recon_
            ref_name_with_prefix = "recon_" + ref_name_no_ext
            # print(ref_name_with_prefix)
            # print(img_name_no_ext)
            if img_name_no_ext == ref_name_with_prefix:
                return ref_file
        
        return None
    
    def _save_results(self, all_similarities: Dict[str, Dict[str, float]], output_file: str):
        """
        Save similarity results to file
        
        Args:
            all_similarities: Dictionary of all models' similarity results
            output_file: Output file path
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data to save
        results = {
            "model_types": self.model_types,
            "similarities_by_model": {},
            "statistics_by_model": {},
            "summary": {}
        }
        
        # Calculate statistics for each model
        all_model_averages = {}
        
        for model_name, similarities in all_similarities.items():
            results["similarities_by_model"][model_name] = similarities
            
            # Calculate statistics for this model
            valid_scores = [score for score in similarities.values() if score is not None]
            if valid_scores:
                model_stats = {
                    "total_images": len(similarities),
                    "valid_pairs": len(valid_scores),
                    "average_similarity": sum(valid_scores) / len(valid_scores),
                    "min_similarity": min(valid_scores),
                    "max_similarity": max(valid_scores)
                }
                results["statistics_by_model"][model_name] = model_stats
                all_model_averages[model_name] = model_stats["average_similarity"]
        
        # Add overall summary
        if all_model_averages:
            results["summary"] = {
                "models_used": list(all_model_averages.keys()),
                "average_scores_by_model": all_model_averages,
                "overall_average": sum(all_model_averages.values()) / len(all_model_averages),
                "best_model": max(all_model_averages.items(), key=lambda x: x[1]),
                "worst_model": min(all_model_averages.items(), key=lambda x: x[1])
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """
    Main function for testing and command line invocation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate image similarity between two folders using multiple models")
    parser.add_argument("--image_path", default="./eval/UniBench/example_image", help="Your image")
    parser.add_argument("--ref_image_path", default="./eval/UniBench/Image", help="Path to the reference image folder")
    parser.add_argument("--output", default="./eval/result/example.json", help="Output file path")
    parser.add_argument("--models", nargs='+', default=["clip", "longclip", "dinov2", "dinov3"], 
                       choices=["clip", "dinov2", "dinov3", "longclip"],
                       help="Models to use for similarity calculation")
    
    args = parser.parse_args()
    
    print(f"Using models: {args.models}")
    
    # Initialize calculator
    calculator = ImageSimilarityCalculator(model_types=args.models)
    
    # Calculate similarity
    all_similarities = calculator.calculate_folder_similarity(
        image_path=args.image_path,
        ref_image_path=args.ref_image_path,
        output_file=args.output
    )
    
    print(f"\n=== FINAL SUMMARY ===")
    for model_name, similarities in all_similarities.items():
        valid_scores = [score for score in similarities.values() if score is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"{model_name.upper()}: Average similarity = {avg_score:.4f} ({len(valid_scores)} valid pairs)")
    
    print(f"\nSimilarity calculation completed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

