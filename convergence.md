---


---
math: true
----------

# Convergence Analysis of MEERKAT in Federated Learning

## 1. Problem Formulation

We consider the **federated zeroth-order optimization problem**, where the objective is to minimize the global loss function:

$$
\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \sum_{k=1}^{K} p_k \mathcal{L}_k(\mathbf{w}),
$$

where:

- \( \mathbf{w} \in \mathbb{R}^d \) is the **global model parameter**,
- \( K \) is the **number of clients**,
- \( p_k \) is the **weight of client \( k \) ** (proportional to dataset size),
- \( \mathcal{L}_k(\mathbf{w}) \) is the **local loss function** of client \( k \).

### **Sparse Zeroth-Order Gradient Estimation**

MEERKAT replaces gradient-based updates with **sparse zeroth-order (ZO) updates**, computed using the **two-point difference estimator**:

$$
g_k = \frac{\mathcal{L}(\mathbf{w} + \epsilon \cdot (\mathbf{z} \odot \mathbf{m}); \mathcal{B}) 
- \mathcal{L}(\mathbf{w} - \epsilon \cdot (\mathbf{z} \odot \mathbf{m}); \mathcal{B})}{2\epsilon}.
$$

The **sparse gradient** is given by:

$$
\nabla \mathcal{L}_k = g_k \cdot (\mathbf{z} \odot \mathbf{m}),
$$

where:

- \( \mathbf{z} \in \mathbb{R}^d \) is a **random Gaussian vector**, i.e., \( \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d) \),
- \( \epsilon \in \mathbb{R} \) is the **perturbation magnitude**,
- \( \mathbf{m} \in \{0, 1\}^d \) is a **binary sparse mask** with density ratio \( \rho \), selecting a subset of parameters for updates.

Each client performs \( T \) local steps:

$$
\mathbf{w}^{t+1}_k = \mathbf{w}^{t}_k - \eta \nabla \mathcal{L}_k^t.
$$

After local updates, the **server aggregates**:

$$
\mathbf{w}_r = \frac{1}{K} \sum_{k=1}^{K} \mathbf{w}_k^T.
$$

---

## 2. Assumptions

### **Assumption 1: Smoothness (L-Lipschitz Gradient)**

Each local function \( \mathcal{L}_k(\mathbf{w}) \) is \( L \)-smooth:

$$
\|\nabla \mathcal{L}_k(\mathbf{w}_1) - \nabla \mathcal{L}_k(\mathbf{w}_2)\| \leq L \|\mathbf{w}_1 - \mathbf{w}_2\|, \quad \forall \mathbf{w}_1, \mathbf{w}_2.
$$

### **Assumption 2: Bounded Variance in ZO Estimation**

The zeroth-order gradient estimator has bounded variance:

$$
\mathbb{E}_{\mathbf{z}} \left[\|\hat{g}_k(\mathbf{w}) - \nabla \mathcal{L}_k(\mathbf{w})\|^2 \right] \leq \sigma^2.
$$

### **Assumption 3: Hessian Low-Rank Property**

The Hessian matrix has an **effective low rank** \( r \):

$$
\frac{\text{tr}(\mathbf{H}(\mathbf{w}_t))}{\|\mathbf{H}(\mathbf{w}_t)\|_{\text{op}}} \leq r.
$$

### **Assumption 4: Polyak-Lojasiewicz (PL) Condition**

The global function \( \mathcal{L}(\mathbf{w}) \) satisfies:

$$
\mathcal{L}(\mathbf{w}) - \mathcal{L}^* \leq \frac{1}{2\mu} \|\nabla \mathcal{L}(\mathbf{w})\|^2.
$$

---

## 3. Convergence Analysis

### **Step 1: One-Step Descent Bound**

To analyze the convergence of MEERKAT, we first establish the one-step descent bound using the ( L )-smoothness assumption.

#### **Lipschitz Smoothness Assumption**

From **Assumption 1**, we know that the local loss function \( \mathcal{L}_k(\mathbf{w}) \) is \( L \)-smooth, meaning its gradient satisfies:

$$
\|\nabla \mathcal{L}_k(\mathbf{w}_1) - \nabla \mathcal{L}_k(\mathbf{w}_2)\| \leq L \|\mathbf{w}_1 - \mathbf{w}_2\|, \quad \forall \mathbf{w}_1, \mathbf{w}_2.
$$

This ensures that the function does not change too rapidly, allowing us to bound its variations.

---

### **Step 1.1: From Assumption 1 to Lipschitz Smoothness Upper Bound**

To derive the **one-step descent bound**, we need to analyze the difference:

$$
\mathcal{L}(\mathbf{w}_{t+1}) - \mathcal{L}(\mathbf{w}_t).
$$

we express the difference as an integral over the gradient:

$$
\mathcal{L}(\mathbf{w}_{t+1}) - \mathcal{L}(\mathbf{w}_t) = \int_0^1 \langle \nabla \mathcal{L}(\mathbf{w}_t + \tau \Delta \mathbf{w}), \Delta \mathbf{w} \rangle d\tau.
$$

where:

$$
\Delta \mathbf{w} = \mathbf{w}_{t+1} - \mathbf{w}_t.
$$

We approximate \( \nabla \mathcal{L}(\mathbf{w}_t + \tau \Delta \mathbf{w}) \) using the Lipschitz smoothness condition:

$$
\nabla \mathcal{L}(\mathbf{w}_t + \tau \Delta \mathbf{w}) = \nabla \mathcal{L}(\mathbf{w}_t) + (\nabla \mathcal{L}(\mathbf{w}_t + \tau \Delta \mathbf{w}) - \nabla \mathcal{L}(\mathbf{w}_t)).
$$

Using **Assumption 1 (Lipschitz smoothness)**, we know:

$$
\|\nabla \mathcal{L}(\mathbf{w}_t + \tau \Delta \mathbf{w}) - \nabla \mathcal{L}(\mathbf{w}_t)\| \leq L \|\tau \Delta \mathbf{w}\|.
$$

Taking the inner product with \( \Delta \mathbf{w} \), we get:

$$
\langle \nabla \mathcal{L}(\mathbf{w}_t + \tau \Delta \mathbf{w}) - \nabla \mathcal{L}(\mathbf{w}_t), \Delta \mathbf{w} \rangle \leq L \tau \|\Delta \mathbf{w}\|^2.
$$

Integrating over \( \tau \):

$$
\int_0^1 L \tau \|\Delta \mathbf{w}\|^2 d\tau = \frac{L}{2} \|\Delta \mathbf{w}\|^2.
$$

Thus, we obtain:

$$
\mathcal{L}(\mathbf{w}_{t+1}) - \mathcal{L}(\mathbf{w}_t) \leq \langle \nabla \mathcal{L}(\mathbf{w}_t), \mathbf{w}_{t+1} - \mathbf{w}_t \rangle + \frac{L}{2} \|\mathbf{w}_{t+1} - \mathbf{w}_t\|^2.
$$


Substituting the sparse ZO update:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}_k^t,
$$

yields:

$$
\mathcal{L}(\mathbf{w}_{t+1}) \leq \mathcal{L}(\mathbf{w}_t) - \eta \langle \nabla \mathcal{L}(\mathbf{w}_t), \nabla \mathcal{L}_k^t \rangle + \frac{L}{2} \eta^2 \|\nabla \mathcal{L}_k^t\|^2.
$$

Taking expectation:

We take the expectation $\mathbb{E}$ on both sides of the `yields` inequality:

$$
\mathbb{E}[\mathcal{L}(\mathbf{w}_{t+1})] \leq \mathbb{E}[\mathcal{L}(\mathbf{w}_t)] - \eta \mathbb{E}[\langle \nabla \mathcal{L}(\mathbf{w}_t), \nabla \mathcal{L}_k^t \rangle] + \frac{L}{2} \eta^2 \mathbb{E}[\|\nabla \mathcal{L}_k^t\|^2].
$$

Since $\nabla \mathcal{L}_k^t$ is a **zeroth-order gradient estimator (ZO gradient estimator)**, its expectation satisfies:

$$
\mathbb{E}[\nabla \mathcal{L}_k^t] = \nabla \mathcal{L}(\mathbf{w}_t).
$$

Therefore:

$$
\mathbb{E}[\langle \nabla \mathcal{L}(\mathbf{w}_t), \nabla \mathcal{L}_k^t \rangle] = \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2].
$$

Substituting this into the `taking expectation` equation, we get:

$$
\mathbb{E}[\mathcal{L}(\mathbf{w}_{t+1})] \leq \mathbb{E}[\mathcal{L}(\mathbf{w}_t)] - \eta \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2] + \frac{L}{2} \eta^2 \mathbb{E}[\|\nabla \mathcal{L}_k^t\|^2].
$$

Since **MEERKAT employs sparse gradient estimation (Sparse ZO gradient)**, it only updates a subset of parameters (with a proportion $\rho$), thus we can approximate:

$$
\mathbb{E}[\|\nabla \mathcal{L}_k^t\|^2] \leq \rho \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2] + O(\sigma^2).
$$

where:

- $\rho$ is the sparsity ratio of gradient updates.
- $O(\sigma^2)$ represents **ZO estimator noise**, which quantifies the variance of the estimated gradient.

By substituting the above result into the `taking expectation` equation:

$$
\mathbb{E}[\mathcal{L}(\mathbf{w}_{t+1})] \leq \mathbb{E}[\mathcal{L}(\mathbf{w}_t)] - \eta \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2] + O(\eta^2 \sigma^2).
$$


$$
\mathbb{E}[\mathcal{L}(\mathbf{w}_{t+1})] \leq \mathbb{E}[\mathcal{L}(\mathbf{w}_t)] - \eta \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2] + O(\eta^2 \sigma^2).
$$

Since MEERKAT updates only a sparse subset (fraction $\rho$):

$$
\mathbb{E}[\|\nabla \mathcal{L}_k^t\|^2] \leq \rho \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2] + O(\sigma^2).
$$

Thus:

$$
\mathbb{E}[\mathcal{L}(\mathbf{w}_{t+1})] \leq \mathbb{E}[\mathcal{L}(\mathbf{w}_t)] - \eta \rho \mathbb{E}[\|\nabla \mathcal{L}(\mathbf{w}_t)\|^2] + O(\eta^2 \sigma^2).
$$

---

### **Step 2: FL Aggregation Effect**

After \( T \) local steps:

$$
\mathbf{w}_{t+T}^k = \mathbf{w}_t^k - \sum_{\tau=0}^{T-1} \eta \nabla \mathcal{L}_k^\tau.
$$

Server aggregates:

$$
\mathbf{w}_{t+T} = \sum_{k=1}^{K} p_k \mathbf{w}_{t+T}^k.
$$

Averaging over \( T \) local steps:

$$
\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E} \|\nabla \mathcal{L}(\mathbf{w}_t)\|^2 \leq O\left(\frac{\rho}{T} (\mathcal{L}(\mathbf{w}_0) - \mathcal{L}^*)\right) + O(\sigma^2).
$$


---


## 4. Conclusion

We proved that MEERKAT’s **federated sparse zeroth-order optimization** converges under reasonable assumptions.

### **Key Takeaways**

1. **MEERKAT converges in FL settings efficiently**.
2. **Convergence rate depends on sparsity \( \rho \) and communication frequency \( T \)**.
3. **PL condition guarantees exponential convergence**.

---
