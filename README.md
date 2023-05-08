# PyTorch-ViT-Model-from-Scratch
Implementation of the popular [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf) model ViT-Base from scratch using layers produced with math from PyTorch tensors (PyTorch Conv layer used out of the box for image patching). The Model matches the original parameters exactly.

Provided is a Notebook with exploration and a .py with the neccessary classes to build the model.


# Tasks to build ViT architecture

- Covert image into patches via learnable embeddings
<p>&nbsp;</p>
- Build transformer encoder layer matching described architecture:
    - x -> LayerNorm -> Multi-Head Attention -> + -> LayerNorm -> MultiLayerPerceptron -> Out
    - Include residual skip connections from input to "+" then from "+" to output 
<p>&nbsp;</p>
- Assemble model
    - Append learnable class embedding to position 0 and add positional embeddings to Transformer input
    - Stack 12 transformer encoder layers sequentially
<img src="https://production-media.paperswithcode.com/models/Screen_Shot_2021-02-14_at_2.26.57_PM_WBwCIco.png" width="667" height="376">


<h1>1: Patch Embeddings</h1>
<p>&nbsp;</p>
The embedding layer for Vision transformers are implemented by takeing the image $$x \in \mathbb{R}^{Height \times Width \times Channels}$$ as an input and segment it into N patches $$x_{p} \in \mathbb{R}^{N \times (Patch ^{2} \cdot Channels)}$$ Then we flatten the patches and use a linear projection to ensure its learnable (embedding) and can be fed into the transformer


<h2>2: Transformer Encoder Layer</h2>

Components required:

- Single headed self-attention mechanism (Needs Linear layer)
- Concatenate multiple heads in parallel for multiheaded self attention
- Layer Norm
- MLP with one hidden layer
- Create Block

<h2>3: Construct Model</h2>

![image](https://user-images.githubusercontent.com/85456951/236855465-140dcff1-cfd4-4a98-b264-c2e14d798836.png)

NOTE!
Its worth noting that ViT included bias' in its transformers linear layers. However current research and evidense show that no bias in the kqv layers leads to better results when training
