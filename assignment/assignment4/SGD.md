

# SGD收敛速率

计算机学院+20020129+王悟信+SGD收敛速率

## 证明目标：

$$
\mathbb{E}_{V_{1:T}}[\frac{1}{T}\sum\limits^T_{i=1}(f(w^{(t)})-f(w^{\star})]\leq\mathbb{E}_{V_{1:T}}[\frac{1}{T}\sum\limits^T_{i=1}\langle{w^{(t)}-w^{\star},v_t}\rangle]
$$

## 证明：

$$
\begin{aligned}
\mathop{\mathbb{E}}\limits_{v_{1:T}}[\frac{1}{T}\sum\limits^T_{i=1}\langle{w^(t)-w^\star,v_t}\rangle]&=\frac{1}{T}\sum^T_{i=1}\mathop{\mathbb{E}}\limits_{v_{1:T}}\langle{w^{(t)}-w^\star,v_t}\rangle
\end{aligned}
\tag{1}\label{eq1}
$$

则$(\ref{eq1})$右侧中取出一项，由于$w^{(t)}=w^{(t-1)}-\eta v_{t-1}$，则对于当前的$t$，$\mathbb{E}$中$i>t$的部分均不用考虑，根据全期望公式：
$$
\forall 变量\alpha,\beta和某个函数g\\
\mathbb{E_\alpha}[g({\alpha})]=\mathbb{E_\beta}[\mathbb{E_\alpha}[g(\alpha)|\beta]]
\tag{2}\label{eq2}
$$
有：
$$
\begin{aligned}
\mathop{\mathbb{E}}\limits_{v_{1:T}}[\langle{w^{(t)}-w^\star,v_t}\rangle]
&=\mathop{\mathbb{E}}\limits_{v_{1:t}}[\langle{w^{(t)}-w^\star,v_t}\rangle]\\
&=\mathop{\mathbb{E}}\limits_{1:t-1}[\mathop{\mathbb{E}}\limits_{1:t}[\langle{w^{(t)}-w^\star,v_t}\rangle|v_{1:t-1}]]
\end{aligned}
\tag{3}\label{eq3}
$$
当$v_{1:t-1}$确定时，$w^{(t)}$也就确定了，所以：
$$
\begin{aligned}
\mathop{\mathbb{E}}\limits_{1:t-1}\mathop{\mathbb{E}}\limits_{1:t}[\langle{w^{(t)}-w^\star,v_t}\rangle|v_{1:t-1}]
\end{aligned}
=\mathop{\mathbb{E}}\limits_{v_{1:t-1}}\langle{w^{(t)}-w^\star,\mathop{\mathbb{E}}\limits_{v_t}}[v_t|v_{t-1}]\rangle
\tag{4}\label{eq4}
$$
由于SGD算法要求$\mathop{\mathbb{E}}\limits_{v_t}[v_t|w^{(t)}]\in\partial f(w^{(t)})$，所以：
$$
\mathop{\mathbb{E}}\limits_{v_{1:t-1}}\langle{w^{(t)}-w^\star,\mathop{\mathbb{E}}\limits_{v_t}[v_t|v_{t-1}]}\rangle
\geq \mathop{\mathbb{E}}\limits_{v_{1:t-1}}[f(w^{(t)})-f(w^\star)]
\tag{5}\label{eq5}
$$
所以
$$
\begin{aligned}
\mathop{\mathbb{E}}\limits_{v_{1:T}}[\langle{w^{(t)-w^\star},v_t}\rangle] &\geq \mathop{\mathbb{E}}\limits_{v_{1:t-1}}[f{w^{(t)}-f(w^\star)}]\\
&=\mathop{\mathbb{E}}\limits_{v_{1:T}}[f{w^{(t)}-f(w^\star)}]
\end{aligned}
\tag{6}\label{eq6}
$$
故要证的式子成立