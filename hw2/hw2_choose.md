# Choose

## 1 D

## 2 E

```python
result[0, 1, 2] = b[1] + (W[1] * x[0, :, 2 * stride:2 * stride + kernel_size]).sum()
```

## 3 A

## 4 D

$$\frac{\partial L}{\partial w_2}=\frac{\partial L}{\partial y_1}\frac{\partial y_1}{\partial w_2}+\frac{\partial L}{\partial y_2}\frac{\partial y_2}{\partial w_2}+\frac{\partial L}{\partial y_3}\frac{\partial y_3}{\partial w_2}=\frac{\partial L}{\partial y_1}x_2+\frac{\partial L}{\partial y_2}x_3+\frac{\partial L}{\partial y_3}x_4$$

$$\frac{\partial L}{\partial x_2}=\frac{\partial L}{\partial y_1}\frac{\partial y_1}{\partial x_2}+\frac{\partial L}{\partial y_2}\frac{\partial y_2}{\partial x_2}=\frac{\partial L}{\partial y_1}w_2+\frac{\partial L}{\partial y_2}w_1$$

## 5 B
