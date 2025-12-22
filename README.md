# Skin Disease Detection - Model Comparison

This project compares various deep learning models for skin disease detection using transfer learning and custom architectures.

## Model Performance Comparison

### Rankings:

1. **YOLOv8** (Best Performer)
   - Accuracy: **90%**
   - Approach: Transfer learning
   - Status: ✅ Recommended

2. **ResNet** (Second Best)
   - Accuracy: **77%**
   - Approach: Transfer learning
   - Status: ✅ Good alternative

### Other Models (Overfitting Issues):

- **CNN (Custom)**
  - Accuracy: ~65-70%
  - Issue: Severe overfitting
  - Status: ⚠️ Not recommended
  - **Transformer**
  - Accuracy: ~72%
  - Issue: Overfitting on validation data
  - Status: ⚠️ Not recommended


## Conclusions

Based on our experiments:
- **YOLOv8** with transfer learning provides the best results with 90% accuracy
- **ResNet** is a solid second choice with 77% accuracy
- Custom CNN and Transformer models struggle with overfitting and should not be used for this task
- **LSTM** achieves 67% accuracy with reasonable generalization and could be used as an alternative
## Repository Structure

- `YOLOv8/` - YOLOv8 implementation (best performing model)
- `RESTNET/` - ResNet transfer learning implementation
- `CNN_MODEL.ipynb` - Custom CNN implementation
- `LSTM_Model.ipynb` - LSTM implementation
- `Transformers.ipynb` - Transformer-based approach
- `dataset/` - Training dataset
