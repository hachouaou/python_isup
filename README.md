Dans le cas d'une régression linéaire multiple, nous disposons de d variables explicatives, notées respectivements $(x^1, ..., x^{d})$, et une variable à expliquer, notée $y$, liées par le modèle suivant :
$$y_i=\beta_0+\beta_1x^1_i+\beta_2x^2_i+...+\beta_{d}x^{d}_i+\varepsilon_i, \quad \text{avec} \quad \beta \in \mathbb{R}^{d+1} \quad \text{et pour tout} \quad 1\le i \le n$$
Que nous pouvous également noter:
$$    y_i = \mathbf{X} \beta + \epsilon_i, \quad \text{pour tout} \quad 1\le i \le n$$
où les $\varepsilon_i$ sont les résidus, nous supposons que $\varepsilon_i$ sont des variables aléatoires centrées de variance $\sigma^2$, et éventuellement de loi normale $\mathcal N(0,\sigma^2)$,
avec:
$$
\mathbf{X} =
\begin{pmatrix}
1 & x^1_1 & ... & x^{d}_1\\
\vdots & \vdots & \vdots & \vdots\\
1 & x^1_n & ... & x^{d}_n
\end{pmatrix} ; \quad \beta =( \beta_0, \beta_1, ..., \beta_{d}) ^\intercal 
$$
L'estimateur $\hat\beta$ obtenu par la méthode des moindres carrés est donné par:
$$\hat\beta = (X^TX)^{-1}X^Ty$$

Une fois les paramètres estimés, on obtient la droite de régression:
$$f(x) = \hat\beta_0+\hat\beta_1x^1_i+\hat\beta_2x^2_i+...+\hat\beta_{d}x^{d}_i,$$
ce qui permet  d'effectuer des prévisions pour des nouvelle variables $x^*=(x^{*1}, ..., x^{*d})$ par: $$y^{pred}=f(x^*)=\hat\beta_0+\hat\beta_1x^{*1}_i+\hat\beta_2x^{*2}_i+...+\hat\beta_{d}x^{*d}_i$$

Les valeurs ajustées sont définies par:
$$\hat y_i=f(x_i)=\hat\beta_0+\hat\beta_1x^1_i+\hat\beta_2x^2_i+...+\hat\beta_{d}x^{d}_i,$$
et les résidus estimés par:
$$\hat\varepsilon_i=y_i-\hat y_i.$$
