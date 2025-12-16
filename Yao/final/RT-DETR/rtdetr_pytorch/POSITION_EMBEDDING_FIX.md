"""
Key fix for position embedding error:
- Process ONE sample at a time instead of batching
- This避免different feature map sizes causing position embedding mismatch

The error "size of tensor a (500) must match size of tensor b (400)" 
happens because RT-DETR's encoder computes position embeddings based on 
feature map size, and batching images with different sizes causes mismatch.

With batch_size=1 and processing single samples, all feature maps will 
have consistent size and position embeddings will work correctly.
"""
