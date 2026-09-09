"""
training 
    python train.py           # both
    python train.py --main    # digits only
    python train.py --stretch # odd/even only

digits: Linear(64,32) -> ReLU -> Linear(32,10) -> Softmax+CE  (SGD and Adam)
stretch: Linear(64,16) -> Tanh -> Linear(16,1) -> Sigmoid+MSE
"""
