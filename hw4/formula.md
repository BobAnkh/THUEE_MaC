# formula

## 1

$$
\mu_{1}=(x_{1}+x_{2}+x_{3})\div3=(-1,\frac{5}{3})^T
$$

$$
\mu_{2}=(x_{4}+x_{5}+x_{6})\div3=(\frac{5}{3},-\frac{2}{3})^T
$$

$$
S_w=\frac{1}{N}\sum_{i=1}^C\sum_{k=1}^{n_i}(x_k^i-\mu_i)(x_k^i-\mu_i)^T=\begin{bmatrix}1.1111 & 0.8889\\0.8889 & 0.8889 \end{bmatrix}
$$

$$
w=S_w^{-1}(\mu_1-\mu_2)=\begin{bmatrix}-22.5000\\25.125 0\end{bmatrix}
$$

## 2

### (1)

$$
\alpha_1(1)=\pi_1b_1(o_1)=0.5\times0.4=0.2
$$

$$
\alpha_1(2)=\pi_2b_2(o_1)=0.5\times0.1=0.05
$$

$$
\alpha_2(1)=[\alpha_1(1)a_{11}+\alpha_1(2)a_{21}]b_1(o_2)=(0.2\times0.7+0.05\times0.2)\times0.1=0.015
$$

$$
\alpha_2(2)=[\alpha_1(1)a_{12}+\alpha_1(2)a_{22}]b_2(o_2)=(0.2\times0.3+0.05\times0.8)\times0.4=0.04
$$

$$
\alpha_3(1)=[\alpha_2(1)a_{11}+\alpha_2(2)a_{21}]b_1(o_2)=(0.015\times0.7+0.04\times0.2)\times0.1=0.00185
$$

$$
\alpha_3(2)=[\alpha_2(2)a_{12}+\alpha_2(2)a_{22}]b_2(o_2)=(0.015\times0.3+0.04\times0.8)\times0.4=0.0146
$$

$$
P(x|\lambda)=\sum_{i=1}^N\alpha_3(i)=0.00185+0.0146=0.01645
$$

### (2)

$$
\delta_1(1)=\pi_1b_1(o_1)=0.5\times0.4=0.2
$$

$$
\delta_1(2)=\pi_2b_2(o_1)=0.5\times0.1=0.05
$$

$$
\Psi_1(1)=0
$$

$$
\Psi_1(2)=0
$$

$$
\delta_2(1)=max_{1\le i \le 2}[\delta_{1}(i)a_{i1}]b_i(o_2)=0.014
$$

$$
\delta_2(2)=max_{1\le i \le 2}[\delta_{1}(i)a_{i2}]b_i(o_2)=0.024
$$

$$
\Psi_2(1)=max_{1\le i \le 2}[\delta_{1}(i)a_{i1}]=1
$$

$$
\Psi_2(2)=max_{1\le i \le 2}[\delta_{1}(i)a_{i2}]=1
$$

$$
\delta_3(1)=max_{1\le i \le 2}[\delta_{2}(i)a_{i1}]b_i(o_2)=0.00098
$$

$$
\delta_3(2)=max_{1\le i \le 2}[\delta_{2}(i)a_{i2}]b_i(o_2)=0.00768
$$

$$
\Psi_3(1)=max_{1\le i \le 2}[\delta_{2}(i)a_{i1}]=1
$$

$$
\Psi_3(2)=max_{1\le i \le 2}[\delta_{2}(i)a_{i2}]=2
$$

$$
P^*=max_{1\le i\le 2}[\delta_3(i)]=0.00768
$$

$$
q_3^*=argmax_{1\le i\le 2}[\delta_3(i)]=2
$$

$$
\{S_1,S_2,S_2\}
$$

## 3

### (1)

$$
g_m(z_1,z_2)=1.99929z_1-8.99692
$$

$$
g_m(z_1,z_2)=1.99929(x_2^2-2x_1+3)-8.99692=1.99929x_2^2-3.99858x_1-2.99905
$$



### (2)

$$
maximize(\sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jK(x_i,x_j))
$$

$$
\alpha_1=0,\alpha_2=0.4857,\alpha_3=0.9215，\alpha_4=0.8887，\alpha_5=0.1503，\alpha_6=0.3682，\alpha_7=0
$$

$$
g_k(X)=\sum_{j\in SV}y_j\alpha_jK(X_j,X)+b=0.8887x_1^2-1.7774x_1+0.6667x_2^2-0.00001614x_2-1.6665
$$

### （4）

$$
Y_{8m}=-1，Y_{8k}=-1，Y_{9m}=-1，Y_{9k}=1
$$

