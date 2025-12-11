
<h2>CAD Representation for Neural Networks </h2>

<h2>Network-friendly Representation </h2>
> [!PDF|] [[2105.09492v2.pdf#page=4&selection=132,0,194,22|2105.09492v2, p.4]]
> > To overcome this challenge, we regularize the dimensions of command sequences. First, for each command, its parameters are stacked into a 16×1 vector, whose elements correspond to the collective parameters of all commands in Table 1 (i.e., pi = [x, y, α, f, r, θ, φ, γ, px, py , pz , s, e1, e2, b, u]). Unused parameters for each command are simply set to be −1. Next, we fix the total number Nc of commands in every CAD model M . This is done by padding the CAD model’s command sequence with the empty command 〈EOS〉 until the sequence length reaches Nc. In practice, we choose Nc = 60, the maximal command sequence length appeared in our training dataset
> 
> 


CAD命令本质上是不规则的，不同命令拥有完全不同的参数，不同CAD模型数量不同造成的操作步骤不同，这种不规则性导致输入神经网络会有极大困难，作者提出一个标准化流程来解这一挑战

<h4>参数向量化:统一所有命令表示</h4>

1. 创建16维向量pi = [x, y, α, f, r, θ, φ, γ, px, py , pz , s, e1, e2, b, u])
2. 处理未使用参数：对于一个特定的命令，他只会用到上述向量的一部份，用不到的未使用的参数统一将其值设置为-1

<h3>序列长度标准化</h3>
为解决模型复杂度不同的问题，作者采用填充策略

1. 固定序列长度：整个数据集设置一个固定的命令序列长度N_c
2. 选择最大长度:   通过分析训练集，最长的命令序列包含60个命令，选择N_c = 60 作为标准长度
3. 填充操作：对于一个给定的CAD模型，如果其命令序列长度少于60，就在序列的末尾添加特殊的空命令`〈EOS〉`​ ，直到序列总长度达到60。`〈EOS〉`是一个标记，表示“序列结束”，其参数向量通常全部填充-1

以上可以保证每一个CAD模型都能表示为$60 \times 16$ 的矩阵


<h2>Embedding</h2>


CAD命令序列本质上是符号的离散数据，但深度学习模型需要连续数值向量作为输入，嵌入

> [!PDF|note] [[2105.09492v2.pdf#page=5&selection=9,0,80,1&color=note|2105.09492v2, p.5]]
> > Similar in spirit to the approach in natural language processing [40], we first project every command Ci onto a common embedding space. Yet, different from words in natural languages, a CAD command Ci = (ti, pi) has two distinct parts: its command type ti and parameters pi. We therefore formulate a different way of computing the embedding of Ci: take it as a sum of three embeddings, that is, e(Ci) = ecmd i + eparam i + epos i ∈ RdE .
> 
> 




