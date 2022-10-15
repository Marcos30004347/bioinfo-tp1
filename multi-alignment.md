# Alinhamento Multiplo

A heurística implementada neste trabalho para executar o alinhamento multiplo de sequências
é baseada em duas estratégias principais:

1. Merging de sequencias já alinhadas.
2. Execução do algortmo de Needleman-Wunsch.

A metodologia da heurística consiste criar uma sequencia global, que será alterada à medida
em que se executa o alinhamento desta mesma sequencia global com as demais.
Tal alteração é feita de forma que, considerando um conjunto de sequências

$$\{ s_i \in S, 1 \leq i \leq k \}$$

cria-se uma sequuência gloabl $s_g$ que é inicializada com o valor de $s_1$, então
itera-se pelas $k-1$ sequências restantes.
Após cada alinhamento entre $s_g$ e $s_i$,  atualiza-se $s_g$ a partir de uma operação
de merging entre o resultado do alinhamento de $s_g$ e $s_i$.
O resultado do alinhamento é dado pelo algoritmo de Needleman-Wunsch.

A operação de merging que é realizada em cada iteração, se dá de forma que tenta-se ao máximo
preservar o estado de $s_g$.
Nesse sentido, a partir dos valores retornados da execução do algoritmo de Needleman-Wunsch, $\tilde{s_g}$
e $\tilde{s_i}$, tenta-se preservar ao máximo o valor de $\tilde{s_g}$ durante o merging.
Para isso, o novo valor da sequência global ($s_g'$) é inicializado com $\tilde{s_g}$, sendo que
apenas os *mismatches* são sobrescritos com as posições correspondentes em $\tilde{s_i}$.

Em linhas gerais, a heurística pode ser reporesentada pelo seguinte pseudo código:

```python
def multi_aligment(S):
  s_g = S[0]

  for s_i in S[1:]:
    _s_g, _s_i = needleman_wunsch(s_g, s_i)
    s_g = merge(_s_g, _s_i)

def merge(s_a, s_b):
    result = s_a
    for c, c_index in result:
        if is_mismatch(c):
            c = s_b[c_index]
```

Para a execução do modo *multi-alignment*, o comando a ser executado deve ser:
```
python3 Bioinfo-TP.py <caminho até o arquivo FASTA> --multi
```