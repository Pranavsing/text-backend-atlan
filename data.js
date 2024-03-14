const models = [
	{
		id: 1,
		name: "Image-to-Text Model",
		category: "Computer Vision",
		modelsCode: `
    import torch
  from torch import einsum, nn
  import torch.nn.functional as F
  from torch.autograd import Function
  import torch.distributed as dist
  
  from einops import rearrange, repeat
  
  # helper functions
  
  def exists(val):
      return val is not None
  
  def default(val, d):
      return val if exists(val) else d
  
  # distributed
  
  def pad_dim_to(t, length, dim = 0):
      pad_length = length - t.shape[dim]
      zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
      return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))
  
  def all_gather_variable_batch(t):
      device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()
  
      size = torch.tensor(t.shape[0], device = device, dtype = torch.long)
      sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
      dist.all_gather(sizes, size)
  
      sizes = torch.stack(sizes)
      max_size = sizes.amax().item()
  
      padded_t = pad_dim_to(t, max_size, dim = 0)
      gathered_tensors = [torch.empty_like(padded_t, device = device, dtype = padded_t.dtype) for i in range(world_size)]
      dist.all_gather(gathered_tensors, padded_t)
  
      gathered_tensor = torch.cat(gathered_tensors)
      seq = torch.arange(max_size, device = device)
  
      mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
      mask = rearrange(mask, 'i j -> (i j)')
  
      gathered_tensor = gathered_tensor[mask]
      sizes = sizes.tolist()
  
      return gathered_tensor, sizes
  
  class AllGather(Function):
      @staticmethod
      def forward(ctx, x):
          assert dist.is_initialized() and dist.get_world_size() > 1
          x, batch_sizes = all_gather_variable_batch(x)
          ctx.batch_sizes = batch_sizes
          return x
  
      @staticmethod
      def backward(ctx, grads):
          batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
          grads_by_rank = grads.split(batch_sizes, dim = 0)
          return grads_by_rank[rank]
  
  all_gather = AllGather.apply
  
  
  # normalization
  # they use layernorm without bias, something that pytorch does not offer
  
  
  class LayerNorm(nn.Module):
      def __init__(self, dim):
          super().__init__()
          self.gamma = nn.Parameter(torch.ones(dim))
          self.register_buffer("beta", torch.zeros(dim))
  
      def forward(self, x):
          return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
  
  # residual
  
  
  class Residual(nn.Module):
      def __init__(self, fn):
          super().__init__()
          self.fn = fn
  
      def forward(self, x, *args, **kwargs):
          return self.fn(x, *args, **kwargs) + x
  
  # to latents
  
  
  class EmbedToLatents(nn.Module):
      def __init__(self, dim, dim_latents):
          super().__init__()
          self.to_latents = nn.Linear(dim, dim_latents, bias=False)
  
      def forward(self, x):
          latents = self.to_latents(x)
          return F.normalize(latents, dim=-1)
  
  # rotary positional embedding
  # https://arxiv.org/abs/2104.09864
  
  
  class RotaryEmbedding(nn.Module):
      def __init__(self, dim):
          super().__init__()
          inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
          self.register_buffer("inv_freq", inv_freq)
  
      def forward(self, max_seq_len, *, device):
          seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
          freqs = einsum("i , j -> i j", seq, self.inv_freq)
          return torch.cat((freqs, freqs), dim=-1)
  
  
  def rotate_half(x):
      x = rearrange(x, "... (j d) -> ... j d", j=2)
      x1, x2 = x.unbind(dim=-2)
      return torch.cat((-x2, x1), dim=-1)
  
  
  def apply_rotary_pos_emb(pos, t):
      return (t * pos.cos()) + (rotate_half(t) * pos.sin())
  
  
  # classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
  # https://arxiv.org/abs/2002.05202
  
  
  class SwiGLU(nn.Module):
      def forward(self, x):
          x, gate = x.chunk(2, dim=-1)
          return F.silu(gate) * x
  
  
  # parallel attention and feedforward with residual
  # discovered by Wang et al + EleutherAI from GPT-J fame
  
  
  class ParallelTransformerBlock(nn.Module):
      def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
          super().__init__()
          self.norm = LayerNorm(dim)
  
          attn_inner_dim = dim_head * heads
          ff_inner_dim = dim * ff_mult
          self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))
  
          self.heads = heads
          self.scale = dim_head**-0.5
          self.rotary_emb = RotaryEmbedding(dim_head)
  
          self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
          self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
  
          self.ff_out = nn.Sequential(
              SwiGLU(),
              nn.Linear(ff_inner_dim, dim, bias=False)
          )
  
          # for caching causal mask and rotary embeddings
  
          self.mask = None
          self.pos_emb = None
  
      def get_mask(self, n, device):
          if self.mask is not None and self.mask.shape[-1] >= n:
              return self.mask[:n, :n].to(device)
  
          mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
          self.mask = mask
          return mask
  
      def get_rotary_embedding(self, n, device):
          if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
              return self.pos_emb[:n].to(device)
  
          pos_emb = self.rotary_emb(n, device=device)
          self.pos_emb = pos_emb
          return pos_emb
  
      def forward(self, x, attn_mask=None):
          """
          einstein notation
          b - batch
          h - heads
          n, i, j - sequence length (base sequence length, source, target)
          d - feature dimension
          """
  
          n, device, h = x.shape[1], x.device, self.heads
  
          # pre layernorm
  
          x = self.norm(x)
  
          # attention queries, keys, values, and feedforward inner
  
          q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
  
          # split heads
          # they use multi-query single-key-value attention, yet another Noam Shazeer paper
          # they found no performance loss past a certain scale, and more efficient decoding obviously
          # https://arxiv.org/abs/1911.02150
  
          q = rearrange(q, "b n (h d) -> b h n d", h=h)
  
          # rotary embeddings
  
          positions = self.get_rotary_embedding(n, device)
          q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
  
          # scale
  
          q = q * self.scale
  
          # similarity
  
          sim = einsum("b h i d, b j d -> b h i j", q, k)
  
          # causal mask
  
          causal_mask = self.get_mask(n, device)
          sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
  
          # extra attention mask - for masking out attention from text CLS token to padding
  
          if exists(attn_mask):
              attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
              sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
  
          # attention
  
          sim = sim - sim.amax(dim=-1, keepdim=True).detach()
          attn = sim.softmax(dim=-1)
  
          # aggregate values
  
          out = einsum("b h i j, b j d -> b h i d", attn, v)
  
          # merge heads
  
          out = rearrange(out, "b h n d -> b n (h d)")
          return self.attn_out(out) + self.ff_out(ff)
  
  # cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward
  
  class CrossAttention(nn.Module):
      def __init__(
          self,
          dim,
          *,
          context_dim=None,
          dim_head=64,
          heads=8,
          parallel_ff=False,
          ff_mult=4,
          norm_context=False
      ):
          super().__init__()
          self.heads = heads
          self.scale = dim_head ** -0.5
          inner_dim = heads * dim_head
          context_dim = default(context_dim, dim)
  
          self.norm = LayerNorm(dim)
          self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()
  
          self.to_q = nn.Linear(dim, inner_dim, bias=False)
          self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
          self.to_out = nn.Linear(inner_dim, dim, bias=False)
  
          # whether to have parallel feedforward
  
          ff_inner_dim = ff_mult * dim
  
          self.ff = nn.Sequential(
              nn.Linear(dim, ff_inner_dim * 2, bias=False),
              SwiGLU(),
              nn.Linear(ff_inner_dim, dim, bias=False)
          ) if parallel_ff else None
  
      def forward(self, x, context):
          """
          einstein notation
          b - batch
          h - heads
          n, i, j - sequence length (base sequence length, source, target)
          d - feature dimension
          """
  
          # pre-layernorm, for queries and context
  
          x = self.norm(x)
          context = self.context_norm(context)
  
          # get queries
  
          q = self.to_q(x)
          q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
  
          # scale
  
          q = q * self.scale
  
          # get key / values
  
          k, v = self.to_kv(context).chunk(2, dim=-1)
  
          # query / key similarity
  
          sim = einsum('b h i d, b j d -> b h i j', q, k)
  
          # attention
  
          sim = sim - sim.amax(dim=-1, keepdim=True)
          attn = sim.softmax(dim=-1)
  
          # aggregate
  
          out = einsum('b h i j, b j d -> b h i d', attn, v)
  
          # merge and combine heads
  
          out = rearrange(out, 'b h n d -> b n (h d)')
          out = self.to_out(out)
  
          # add parallel feedforward (for multimodal layers)
  
          if exists(self.ff):
              out = out + self.ff(x)
  
          return out
  
  # transformer
  
  
  class CoCa(nn.Module):
      def __init__(
          self,
          *,
          dim,
          num_tokens,
          unimodal_depth,
          multimodal_depth,
          dim_latents = None,
          image_dim = None,
          num_img_queries=256,
          dim_head=64,
          heads=8,
          ff_mult=4,
          img_encoder=None,
          caption_loss_weight=1.,
          contrastive_loss_weight=1.,
          pad_id=0
      ):
          super().__init__()
          self.dim = dim
  
          self.pad_id = pad_id
          self.caption_loss_weight = caption_loss_weight
          self.contrastive_loss_weight = contrastive_loss_weight
  
          # token embeddings
  
          self.token_emb = nn.Embedding(num_tokens, dim)
          self.text_cls_token = nn.Parameter(torch.randn(dim))
  
          # image encoder
  
          self.img_encoder = img_encoder
  
          # attention pooling for image tokens
  
          self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
          self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)
  
          self.img_attn_pool_norm = LayerNorm(dim)
          self.text_cls_norm = LayerNorm(dim)
  
          # to latents
  
          dim_latents = default(dim_latents, dim)
          self.img_to_latents = EmbedToLatents(dim, dim_latents)
          self.text_to_latents = EmbedToLatents(dim, dim_latents)
  
          # contrastive learning temperature
  
          self.temperature = nn.Parameter(torch.Tensor([1.]))
  
          # unimodal layers
  
          self.unimodal_layers = nn.ModuleList([])
          for ind in range(unimodal_depth):
              self.unimodal_layers.append(
                  Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
              )
  
          # multimodal layers
  
          self.multimodal_layers = nn.ModuleList([])
          for ind in range(multimodal_depth):
              self.multimodal_layers.append(nn.ModuleList([
                  Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                  Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
              ]))
  
          # to logits
  
          self.to_logits = nn.Sequential(
              LayerNorm(dim),
              nn.Linear(dim, num_tokens, bias=False)
          )
  
          # they used embedding weight tied projection out to logits, not common, but works
          self.to_logits[-1].weight = self.token_emb.weight
          nn.init.normal_(self.token_emb.weight, std=0.02)
  
          # whether in data parallel setting
          self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
  
      def embed_text(self, text):
          batch, device = text.shape[0], text.device
  
          seq = text.shape[1]
  
          text_tokens = self.token_emb(text)
  
          # append text cls tokens
  
          text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
          text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)
  
          # create specific mask for text cls token at the end
          # to prevent it from attending to padding
  
          cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')
          attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)
  
          # go through unimodal layers
  
          for attn_ff in self.unimodal_layers:
              text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)
  
          # get text cls token
  
          text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
          text_embeds = self.text_cls_norm(text_cls_tokens)
          return text_embeds, text_tokens
  
      def embed_image(self, images=None, image_tokens=None):
          # encode images into embeddings
          # with the img_encoder passed in at init
          # it can also accept precomputed image tokens
  
          assert not (exists(images) and exists(image_tokens))
  
          if exists(images):
              assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
              image_tokens = self.img_encoder(images)
  
          # attention pool image tokens
  
          img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
          img_queries = self.img_attn_pool(img_queries, image_tokens)
          img_queries = self.img_attn_pool_norm(img_queries)
  
          return img_queries[:, 0], img_queries[:, 1:]
  
      def forward(
          self,
          text,
          images=None,
          image_tokens=None,
          labels=None,
          return_loss=False,
          return_embeddings=False
      ):
          batch, device = text.shape[0], text.device
  
          if return_loss and not exists(labels):
              text, labels = text[:, :-1], text[:, 1:]
  
          text_embeds, text_tokens = self.embed_text(text)
  
          image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)
  
          # return embeddings if that is what the researcher wants
  
          if return_embeddings:
              return text_embeds, image_embeds
  
          # go through multimodal layers
  
          for attn_ff, cross_attn in self.multimodal_layers:
              text_tokens = attn_ff(text_tokens)
              text_tokens = cross_attn(text_tokens, image_tokens)
  
          logits = self.to_logits(text_tokens)
  
          if not return_loss:
              return logits
  
          # shorthand
  
          ce = F.cross_entropy
  
          # calculate caption loss (cross entropy loss)
  
          logits = rearrange(logits, 'b n c -> b c n')
          caption_loss = ce(logits, labels, ignore_index=self.pad_id)
          caption_loss = caption_loss * self.caption_loss_weight
  
          # embedding to latents
  
          text_latents = self.text_to_latents(text_embeds)
          image_latents = self.img_to_latents(image_embeds)
  
          # maybe distributed all gather
  
          if self.is_distributed:
              latents = torch.stack((text_latents, image_latents), dim = 1)
              latents = all_gather(latents)
              text_latents, image_latents = latents.unbind(dim = 1)
  
          # calculate contrastive loss
  
          sim = einsum('i d, j d -> i j', text_latents, image_latents)
          sim = sim * self.temperature.exp()
          contrastive_labels = torch.arange(batch, device=device)
  
          contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
          contrastive_loss = contrastive_loss * self.contrastive_loss_weight
  
          return caption_loss + contrastive_loss
  
  
    `,
		description: `<div>
    <br></br>
    <div className="he">
      <h1>Image-to-Text Model Description</h1>
    </div>
    <br></br>
    <h3>1. Data Collection and Preprocessing:</h3>
    <ul>
      <li>
        <strong>Image Data:</strong> Gather a dataset consisting of paired
        images and corresponding textual descriptions. This dataset is crucial
        for training the model.
      </li>
      <li>
        <strong>Text Data:</strong> Each image should have a detailed and
        relevant textual description. It's common to use datasets like MSCOCO
        or Flickr30k for this purpose.
      </li>
      <li>
        <strong>Preprocessing:</strong> Resize images to a standard size,
        normalize pixel values, and tokenize textual descriptions into a
        format suitable for model training.
      </li>
    </ul>

    <h3>2. Model Architecture:</h3>
    <ul>
      <li>
        <strong>Encoder-Decoder Architecture:</strong> Use a neural network
        architecture with an encoder-decoder structure. The encoder processes
        the image, while the decoder generates the textual description.
      </li>
      <li>
        <strong>Pre-trained Convolutional Neural Network (CNN):</strong>{" "}
        Employ a pre-trained CNN (e.g., ResNet, VGG) as the image encoder. The
        CNN extracts meaningful features from the input image.
      </li>
      <li>
        <strong>
          Recurrent Neural Network (RNN) or Transformer Decoder:
        </strong>{" "}
        Use an RNN (LSTM or GRU) or a Transformer-based model as the text
        decoder. These architectures capture sequential dependencies in the
        generated text.
      </li>
    </ul>

    <h3>3. Training:</h3>
    <ul>
      <li>
        <strong>Loss Function:</strong> Define a suitable loss function, such
        as cross-entropy loss, to measure the difference between predicted and
        actual captions.
      </li>
      <li>
        <strong>Optimization Algorithm:</strong> Use an optimization algorithm
        (e.g., Adam) to minimize the loss during training.
      </li>
      <li>
        <strong>Teacher Forcing:</strong> During training, employ teacher
        forcing, where the true caption is used as input to help the model
        learn the mapping between images and captions.
      </li>
    </ul>

    <h3>4. Transfer Learning:</h3>
    <ul>
      <li>
        <strong>Fine-tuning:</strong> If computational resources are limited,
        leverage transfer learning by fine-tuning a pre-trained image-to-text
        model on your specific dataset. This can accelerate training and
        improve performance.
      </li>
    </ul>

    <h3>5. Evaluation:</h3>
    <ul>
      <li>
        <strong>Metrics:</strong> Measure the model's performance using
        evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE. These
        metrics compare the generated captions to reference captions in the
        test set.
      </li>
      <li>
        <strong>Validation Set:</strong> Use a validation set to monitor the
        model's performance during training and prevent overfitting.
      </li>
    </ul>

    <h3>6. Inference:</h3>
    <ul>
      <li>
        <strong>Greedy Decoding or Beam Search:</strong> During inference, use
        a decoding strategy like greedy decoding or beam search to generate
        captions for new, unseen images.
      </li>
      <li>
        <strong>Post-processing:</strong> Clean up generated captions by
        removing unnecessary tokens and ensuring grammatical correctness.
      </li>
    </ul>

    <h3>7. Deployment:</h3>
    <ul>
      <li>
        <strong>Integration:</strong> Once the model is trained and validated,
        integrate it into applications or services where image-to-text
        functionality is required.
      </li>
      <li>
        <strong>Scalability:</strong> Ensure the model is scalable and can
        handle a diverse range of images.
      </li>
    </ul>

    <h3>8. Continuous Improvement:</h3>
    <ul>
      <li>
        <strong>Feedback Loop:</strong> Establish a feedback loop for
        continuous improvement by collecting user feedback on generated
        captions and updating the model accordingly.
      </li>
    </ul>
  </div>`,
		scenario: `<div>
  <br></br>
  <div className="he">
    <h1 className="he">Assistive Technology for the Visually Impaired</h1>
  </div>

  <br></br>
  <h2>Problem Statement</h2>
  <p>
    Visually impaired individuals face challenges in understanding the
    visual content of images. This makes it difficult for them to access
    information in a world that often relies heavily on visual
    communication.
  </p>

  <h2>Use Case: Image-to-Text Model</h2>
  <p>
    An image-to-text model can be employed to create an assistive technology
    solution. Here's how it works:
  </p>
  <ul>
    <li>
      A visually impaired person uses a mobile app designed for this
      purpose.
    </li>
    <li>The user takes a picture with their smartphone camera.</li>
    <li>The image is sent to an image-to-text model.</li>
    <li>The model generates a descriptive text caption for the image.</li>
    <li>
      Text-to-speech technology converts the generated caption into audible
      feedback for the user.
    </li>
  </ul>

  <h2>Benefits</h2>
  <ul>
    <li>
      <strong>Enables a better understanding of surroundings:</strong>{" "}
      Visually impaired individuals can gain more information about their
      environment through the generated descriptions.
    </li>
    <li>
      <strong>Improved accessibility to information:</strong> The
      image-to-text model can bridge the gap by providing access to
      information presented in visual formats like printed materials, signs,
      or other visual content.
    </li>
  </ul>
  <br></br>
  <br></br>
  <br></br>
  <div className="he">
    <h1 className="he">Content Moderation in Social Media</h1>
  </div>
  <br></br>
  <h2>Problem Statement</h2>
  <p>
    Social media platforms contend with a massive volume of user-generated
    content, making it challenging to ensure it adheres to their community
    guidelines. This includes identifying and removing inappropriate or
    offensive images that can negatively impact user experience.
  </p>

  <h2>Use Case: Image-to-Text Model for Content Moderation</h2>
  <p>
    Machine learning can be leveraged to streamline content moderation.
    Here's how an image-to-text model can be employed:
  </p>

  <ul>
    <li>
      <strong>Image Analysis with Text Descriptions:</strong> The
      image-to-text model analyzes images uploaded by users and generates
      textual descriptions of their content.
    </li>
    <li>
      <strong>Natural Language Processing (NLP):</strong> These textual
      descriptions are then processed by natural language processing (NLP)
      algorithms. NLP techniques can identify keywords, phrases, or overall
      sentiment that might indicate potentially inappropriate content.
    </li>
    <li>
      <strong>Automated Flagging:</strong> If the generated text raises red
      flags based on the NLP analysis, the corresponding image is
      automatically flagged for further review by human moderators.
    </li>
  </ul>

  <h2>Benefits</h2>

  <ul>
    <li>
      <strong>Efficient Workload Management:</strong> By automating the
      initial screening process, the image-to-text model helps reduce the
      burden on human moderators, allowing them to focus on complex cases
      requiring nuanced judgment.
    </li>
    <li>
      <strong>Enhanced Content Filtering:</strong> The image-to-text model,
      combined with NLP analysis, provides a more efficient way to filter
      out inappropriate or offensive content. It can identify potential
      issues that might be missed by solely relying on image recognition
      techniques.
    </li>
  </ul>

  <p>
    It's important to note that this technology should be used as a tool to
    assist human moderators, not replace them entirely. Human expertise
    remains crucial for making final decisions and ensuring fair and
    accurate content moderation.
  </p>
</div>`,
	},
	{
		id: 2,
		name: "Language Translation Model",
		category: "Natural Language Processing",
		modelsCode: `import tkinter
        import customtkinter as ctk
        from PIL import ImageTk
        import torch
        from torch import autocast
        from diffusers import StableDiffusionPipeline 
        
        ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"
        
        class App(ctk.CTk):
            def __init__(self):
                super().__init__()
                # configures window
                self.default_window_width = 1200
                self.default_window_height = 800
                self.authorization_token = ""
        
                self.title("Image Generator")
                self.geometry(f"{self.default_window_width}x{self.default_window_height}")
        
                # generates user interface
                self.windowlabel = ctk.CTkLabel(self,text="Avishake Adhikary's Image Generator", font=ctk.CTkFont(size=30, weight="bold"),padx=50, pady=50,text_color="white")
                self.windowlabel.pack()
                self.promptlabel = ctk.CTkLabel(self,text="Prompt", font=ctk.CTkFont(family="Times New Roman",size=20, weight="bold"),text_color="white")
                self.promptlabel.pack()
                self.promptentry = ctk.CTkEntry(self, placeholder_text="Enter your prompt here",width=self.default_window_width-20, height=40)
                self.promptentry.pack(padx=20, pady=20)
        
                self.generatebutton = ctk.CTkButton(master=self,text="Generate Image",width=self.default_window_width-50, height=40,fg_color="transparent", border_width=2, text_color="white",command=self.generate) 
                self.generatebutton.pack()
        
            def generate(self):
                self.textprompt = self.promptentry.get()
        
                self.generatebutton.configure(state="disabled")
                
                self.progress = ctk.CTkProgressBar(master=self,orientation='horizontal',mode='indeterminate')
                self.progress.pack()
                self.progress.start()
        
                self.modelid = "CompVis/stable-diffusion-v1-4"
                self.device = torch.device("cuda")
                self.pipe = StableDiffusionPipeline.from_pretrained(self.modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=self.authorization_token) 
                self.pipe.to(self.device) 
        
                with autocast():
                    self.image = self.pipe(self.textprompt, guidance_scale=8.5).images[0]
                    self.image.save('generatedimage.png')
                    self.img = ImageTk.PhotoImage(self.image)
                    
                    self.imageview = ctk.CTkLabel(self,width=600,height=400)
                    self.imageview.pack()
                    self.imageview.configure(image=self.img)
        
                self.progress.stop()
                self.progress.pack_forget()
                self.generatebutton.configure(state="normal")
        
        if __name__ == "__main__":
            app = App()
            app.mainloop()
        
        
          `,
		description: `<div>
          <br></br>
          <div className="he">
            <h1 className="he">Language Translation Model</h1>
          </div>
          <br></br>
          <p>
            A language translation model using machine learning (ML) is designed to
            automatically translate text or speech from one language to another.
            This type of model is particularly valuable for breaking down language
            barriers and facilitating communication across diverse linguistic
            communities.
          </p>
    
          <h2>Components and Processes</h2>
    
          <h3>1. Data Collection and Preprocessing</h3>
    
          <ul>
            <li>
              <strong>Parallel Text Corpora:</strong> Gather a large dataset
              containing parallel texts in the source and target languages. These
              texts should be translations of each other.
            </li>
            <li>
              <strong>Sentence Alignment:</strong> Align corresponding sentences in
              the source and target languages to create training pairs.
            </li>
            <li>
              <strong>Tokenization and Normalization:</strong> Tokenize sentences
              into words or subword units and normalize the text by converting it to
              lowercase, removing punctuation, etc.
            </li>
          </ul>
    
          <h3>2. Model Architecture</h3>
    
          <ul>
            <li>
              <strong>Sequence-to-Sequence (Seq2Seq) Architecture:</strong> Utilize
              a neural network model with a Seq2Seq architecture, comprising an
              encoder and a decoder.
            </li>
            <li>
              <strong>Recurrent Neural Network (RNN), LSTM, or Transformer:</strong>{" "}
              Choose an appropriate architecture for the encoder and decoder. While
              RNNs and LSTMs were traditionally used, Transformer models have shown
              superior performance in recent years, especially for long-range
              dependencies.
            </li>
          </ul>
    
          <h3>3. Training</h3>
    
          <ul>
            <li>
              <strong>Loss Function:</strong> Define a suitable loss function, often
              cross-entropy loss, to measure the dissimilarity between predicted and
              actual translations.
            </li>
            <li>
              <strong>Optimization Algorithm:</strong> Use an optimization algorithm
              like Adam or SGD to minimize the loss during training.
            </li>
            <li>
              <strong>Teacher Forcing:</strong> During training, implement teacher
              forcing, where the correct target sequence is used as input to assist
              the model in learning the mapping between languages.
            </li>
          </ul>
    
          <h3>4. Embeddings and Attention Mechanism</h3>
    
          <ul>
            <li>
              <strong>Word Embeddings:</strong> Represent words in a continuous
              vector space using embeddings, which capture semantic relationships
              between words.
            </li>
            <li>
              <strong>Attention Mechanism:</strong> Incorporate attention
              mechanisms, such as in the Transformer model, to enable the model to
              focus on different parts of the source sentence when generating each
              word in the target sentence.
            </li>
          </ul>
    
          <h3>5. Transfer Learning</h3>
    
          <p>
            <strong>Pre-trained Models:</strong> Leverage pre-trained language
            models or embeddings to enhance the performance of the translation
            model, especially if a vast amount of labeled data is not available.
          </p>
    
          <h3>6. Evaluation</h3>
    
          <ul>
            <li>
              <strong>Metrics:</strong> Assess the model's performance using
              evaluation metrics like BLEU (Bilingual Evaluation Understudy),
              METEOR, or TER (Translation Edit Rate). These metrics compare the
              generated translations to reference translations.
            </li>
          </ul>
    
          <h3>7. Inference</h3>
    
          <ul>
            <li>
              <strong>Beam Search or Greedy Decoding:</strong> Use beam search or
              greedy decoding during the inference phase to generate translations
              for new input sentences.
            </li>
            <li>
              <strong>Post-processing:</strong> Apply post-processing techniques to
              improve the fluency and coherence of the generated translations.
            </li>
          </ul>
    
          <h3>8. Deployment</h3>
    
          <ul>
            <li>
              <strong>Integration:</strong> Integrate the trained model into
              applications, websites, or services where language translation
              functionality is required.
            </li>
            <li>
              <strong>Scalability:</strong> Ensure that the model can handle a
              diverse range of input sentences and is scalable to accommodate
              varying translation requirements.
            </li>
          </ul>
    
          <h3>9. Continuous Improvement</h3>
    
          <p>
            <strong>Fine-tuning:</strong> Implement a feedback loop for continuous
            improvement by collecting user feedback on translations and periodically
            fine-tuning the model.
          </p>
    
          <p>
            Building a language translation model involves addressing challenges
            such as handling different sentence lengths, capturing context, and
            managing rare or out-of-vocabulary words. As with other machine learning
            models, the success of a language translation model depends on the
            quality and quantity of the training data, the model architecture, and
            effective training strategies.
          </p>
        </div>`,
		scenario: `<div>
        <br></br>
        <div className="he">
          <h1 className="he">Cross-Language Communication in Customer Support</h1>
        </div>
        <br></br>
        <h2>Problem Statement</h2>
        <p>
          Many businesses operate globally, catering to customers with diverse
          linguistic backgrounds. This can create a communication gap between
          customers and support teams, hindering efficient problem-solving and
          reducing customer satisfaction.
        </p>
  
        <h2>ML Solution: Language Translation Model</h2>
        <p>
          A machine learning (ML) solution can bridge the language gap and
          facilitate seamless communication. Here's how it works:
        </p>
        <ul>
          <li>
            A customer submits a support ticket or query in their preferred
            language.
          </li>
          <li>
            A language translation model automatically translates the customer's
            message into the language preferred by the support team.
          </li>
          <li>
            The support team can then address the issue and provide a solution
            efficiently.
          </li>
        </ul>
  
        <h2>Benefits</h2>
  
        <h3>Improved Responsiveness</h3>
        <p>
          By eliminating language barriers, the translation model allows support
          teams to respond to customer queries faster, improving overall
          responsiveness and customer satisfaction.
        </p>
  
        <h3>Enhanced Customer Experience</h3>
        <p>
          Customers receive support in their native language, leading to a more
          positive and comfortable experience. They can clearly communicate their
          concerns and understand the solutions provided by the support team.
        </p>
  
        <h3>Cost-Efficient</h3>
        <p>
          The language translation model offers a cost-effective alternative to
          hiring multilingual support staff. It can handle a wide range of
          languages, reducing the need for extensive staff training and resource
          allocation.
        </p>
        <br></br>
        <br></br>
        <br></br>
        <div className="he">
          <h1 className="he">Global Content Localization for E-Commerce</h1>
        </div>
        <br></br>
        <h2>Problem Statement</h2>
        <p>
          E-commerce businesses aiming for international success face the
          challenge of making their products accessible and appealing to a global
          audience. This requires adapting product listings and other content to
          resonate with users in different regions, often speaking diverse
          languages.
        </p>
  
        <h2>ML Solution: Language Translation Model</h2>
        <p>
          Machine learning offers a powerful solution for overcoming language
          barriers in e-commerce. Here's how a language translation model can be
          employed:
        </p>
  
        <ul>
          <li>
            The model takes product descriptions, reviews, and other website
            content written in the source language (e.g., English).
          </li>
          <li>
            It translates the content into the target languages spoken by the
            e-commerce platform's target audience in different regions.
          </li>
        </ul>
  
        <h2>Benefits</h2>
  
        <h3>Market Expansion</h3>
        <p>
          By providing product information in the local language, the language
          translation model removes a significant barrier to entry for new
          markets. Customers can easily understand product details, increasing the
          platform's reach and potential customer base.
        </p>
  
        <h3>Increased Sales</h3>
        <p>
          Presenting products in a language familiar to the customer fosters
          better engagement and understanding. Customers are more likely to trust
          and purchase products when descriptions resonate with their cultural
          context and preferences, leading to increased sales.
        </p>
  
        <h3>Consistent Branding</h3>
        <p>
          The language translation model can be fine-tuned to ensure consistent
          brand messaging is conveyed across different languages. This maintains
          the integrity of the brand identity and ensures a cohesive customer
          experience regardless of location.
        </p>
      </div>`,
	},
	{
		id: 3,
		name: "Image Generator Model",
		category: "Image Processing Processing",
		modelCodes: `import pandas as pd
        import re
        import nltk
        from nltk.translate import AlignedSent, IBMModel1
        # Load and Preprocess the Data
        df = pd.read_csv("./engspn.csv")
        english_sentences = df['english'].tolist()
        spanish_sentences = df['spanish'].tolist()
        def clean_sentences(sentences):
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                sentence = sentence.lower()
                sentence = re.sub(r"[^a-zA-Z0-9]+", " ", sentence)
                cleaned_sentences.append(sentence.strip())
            return cleaned_sentences
        cleaned_english_sentences = clean_sentences(english_sentences)
        cleaned_spanish_sentences = clean_sentences(spanish_sentences)
        # Train the Translation Model
        def train_translation_model(source_sentences, target_sentences):
            aligned_sentences = [AlignedSent(source.split(), target.split()) for source, target in zip(source_sentences, target_sentences)]
            ibm_model = IBMModel1(aligned_sentences, 10)
            return ibm_model
        translation_model = train_translation_model(cleaned_english_sentences, cleaned_spanish_sentences)
        # Translate Input Sentences
        def translate_input(ibm_model):
            while True:
                source_text = input("Enter the English sentence to translate (or 'q' to quit): ")
                if source_text.lower() == 'q':
                    print("Quitting...")
                    break
                cleaned_text = clean_sentences(source_text.split())
                source_words = cleaned_text
                translated_words = []
                for source_word in source_words:
                    max_prob = 0.0
                    translated_word = None
                    for target_word in ibm_model.translation_table[source_word]:
                        prob = ibm_model.translation_table[source_word][target_word]
                        if prob > max_prob:
                            max_prob = prob
                            translated_word = target_word
                    if translated_word is not None:
                        translated_words.append(translated_word)
                translated_text = ' '.join(translated_words)
                print("Translated text:", translated_text)
                print()
        translate_input(translation_model)
       
      
               
      
        `,
		description: `<div>
        <br></br>

        <div className="he">
          <h1 className="he">Image Generation Using Text Model</h1>
        </div>
        <br></br>
        <p>
          An image generation using text model, often referred to as a
          text-to-image synthesis model, is a type of machine learning (ML) model
          designed to create realistic images based on textual descriptions. This
          technology is commonly associated with generative models, particularly
          Generative Adversarial Networks (GANs) and Variational Autoencoders
          (VAEs).
        </p>
  
        <h2>Components and Processes</h2>
  
        <h3>1. Data Collection and Preprocessing</h3>
  
        <ul>
          <li>
            <strong>Text Data:</strong> Gather a dataset containing pairs of
            textual descriptions and corresponding images. Ensure that the text is
            detailed and provides sufficient information for image generation.
          </li>
          <li>
            <strong>Image Data:</strong> Preprocess images to a standard size,
            normalize pixel values, and potentially use data augmentation
            techniques to increase the diversity of the dataset.
          </li>
        </ul>
  
        <h3>2. Model Architecture</h3>
  
        <ul>
          <li>
            <strong>Conditional GANs or VAEs:</strong> Use a conditional GAN or
            VAE architecture to enable the generation of images based on input
            text.
          </li>
          <li>
            <strong>Encoder-Decoder Structure:</strong> The model typically
            consists of an encoder that processes the textual input and a decoder
            that generates the corresponding image.
          </li>
          <li>
            <strong>Attention Mechanism:</strong> Integrate attention mechanisms
            to allow the model to focus on specific parts of the textual input
            during the generation process.
          </li>
        </ul>
  
        <h3>3. Text Embedding</h3>
  
        <ul>
          <li>
            <strong>Word Embeddings:</strong> Convert words in the textual
            descriptions into continuous vector representations (word embeddings)
            using techniques like Word2Vec, GloVe, or embeddings layers in neural
            networks.
          </li>
          <li>
            <strong>Sentence Embeddings:</strong> Aggregate word embeddings to
            obtain a fixed-size vector representation of the entire textual
            description, capturing the semantic meaning.
          </li>
        </ul>
  
        <h3>4. Model Training</h3>
  
        <ul>
          <li>
            <strong>Adversarial Training (GANs):</strong> In the case of GANs,
            train the generator to produce realistic images and the discriminator
            to differentiate between real and generated images.
          </li>
          <li>
            <strong>Reconstruction Loss (VAEs):</strong> For VAEs, use a
            combination of a reconstruction loss and a variational loss to ensure
            that the generated images are faithful to the input textual
            descriptions.
          </li>
        </ul>
  
        <h3>5. Transfer Learning</h3>
  
        <p>
          <strong>Pre-trained Models:</strong> Leverage pre-trained models for
          both the text and image components to speed up training and improve
          performance.
        </p>
  
        <h3>6. Evaluation</h3>
  
        <ul>
          <li>
            <strong>Perceptual Metrics:</strong> Assess the visual quality of
            generated images using perceptual metrics like Inception Score or
            Frechet Inception Distance (FID).
          </li>
          <li>
            <strong>User Studies:</strong> Conduct user studies to gather
            subjective feedback on the perceived quality and relevance of
            generated images.
          </li>
        </ul>
  
        <h3>7. Inference</h3>
  
        <ul>
          <li>
            <strong>Sampling Techniques:</strong> During inference, use sampling
            techniques such as greedy decoding or beam search to generate diverse
            sets of images corresponding to a given textual input.
          </li>
          <li>
            <strong>Post-processing:</strong> Fine-tune generated images or
            perform post-processing to enhance visual quality and coherence.
          </li>
        </ul>
  
        <h3>8. Deployment</h3>
  
        <ul>
          <li>
            <strong>Integration:</strong> Once the model is trained and validated,
            integrate it into applications or services where image generation
            based on text descriptions is required.
          </li>
          <li>
            <strong>Scalability:</strong> Ensure the model is scalable to handle a
            variety of textual inputs and can generate high-quality images in
            real-time.
          </li>
        </ul>
  
        <h3>9. Continuous Improvement</h3>
  
        <p>
          <strong>Feedback Loop:</strong> Establish a feedback loop for continuous
          improvement by collecting user feedback on generated images and updating
          the model accordingly.
        </p>
  
        <p>
          Building an image generation using text model involves a combination of
          natural language processing and computer vision techniques, and it is
          crucial to strike a balance between the richness of textual input and
          the capacity of the model to capture and reproduce diverse visual
          content.
        </p>
      </div>`,
		scenario: `<div>
      <br></br>
      <div className="he">
        <h1 className="he">Fashion Design Conceptualization</h1>
      </div>
      <br></br>
      <h2>Use Case: Text-to-Image Synthesis Model</h2>
      <p>
        The initial stages of fashion design often involve brainstorming and
        creating written descriptions for new clothing ideas. A text-to-image
        synthesis model can be a valuable tool in this conceptualization phase,
        transforming textual descriptions into visual representations to inspire
        and guide the design process.
      </p>

      <h3>How it Works</h3>

      <ul>
        <li>
          <strong>Designers Input Text Descriptions:</strong> Designers begin by
          outlining their initial ideas in text format. This could be a detailed
          description of a garment, such as "a flowing maxi dress with a floral
          print, ruffles on the sleeves, and a plunging neckline." They can also
          include keywords that capture the overall mood or style, like
          "romantic," "bohemian," or "vintage."
        </li>
        <li>
          <strong>Model Generates Images:</strong> The text-to-image synthesis
          model takes the designer's textual input and utilizes its machine
          learning capabilities to generate corresponding images.
        </li>
        <li>
          <strong>Visualizing Design Concepts:</strong> The generated images
          provide a visual starting point for the designer's ideas. They can see
          their textual descriptions come to life and explore various
          interpretations of the concept.
        </li>
        <li>
          <strong>Iterative Refinement:</strong> Designers can refine their
          ideas by providing further textual descriptions or modifying existing
          keywords. With each iteration, the model generates new images,
          allowing them to explore different variations and achieve the desired
          visual representation for their fashion design.
        </li>
      </ul>

      <p>
        This technology empowers designers to bridge the gap between their
        creative vision and visual output, fostering a more efficient and
        inspiring conceptualization process.
      </p>
      <br></br>
      <br></br>
      <br></br>
      <div className="he">
        <h1 className="he">Conceptual Art Generation</h1>
      </div>
      <br></br>
      <h2>Use Case: Text-to-Image Synthesis for Art</h2>
      <p>
        Conceptual art and graphic design often involve capturing abstract ideas
        or themes in visual forms. A text-to-image synthesis model can be a
        valuable tool in this creative process, transforming textual
        descriptions into unique and inspiring artworks.
      </p>

      <h3>How it Works</h3>

      <ul>
        <li>
          <strong>Artists Describe Concepts in Text:</strong> Artists begin by
          outlining their ideas in writing. This could be a description of a
          specific scene, an exploration of an emotion, or a representation of a
          symbolic theme. For instance, an artist might describe "a cityscape
          pulsating with neon lights and bustling with energy."
        </li>
        <li>
          <strong>Model Generates Visual Art:</strong> The text-to-image
          synthesis model leverages its machine learning capabilities to
          translate the artist's textual description into a corresponding image.
        </li>
        <li>
          <strong>Inspiration and Exploration:</strong> The generated image
          serves as a visual starting point for the artist's creative
          exploration. It can spark new ideas, provide unexpected
          interpretations, or offer a foundation for further artistic
          development.
        </li>
      </ul>

      <p>
        This technology bridges the gap between verbal concepts and visual
        representations, fostering a more iterative and innovative approach to
        conceptual art creation.
      </p>
    </div>`,
	},
	// Add more models...
];

const featuredModels = [
	{ id: 1, name: "Image-to-Text Model", reason: "Most viewed" },
	// Add more featured models...
];
module.exports = { models, featuredModels };
