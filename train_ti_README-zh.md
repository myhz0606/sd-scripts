[Textual Inversion](https://textual-inversion.github.io/) 学习说明。

[共用学习文档](./train_README-zh.md) 参考。

在实现它时，我曾经参考 https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion 

经过训练的模型可以按直接在 Web UI 中使用。

# 步骤

请事先参考README并进行环境准备。

## 数据准备
请参阅有关准备培训数据的信息。

请参阅有关[训练数据准备的信息](./train_README-zh.md) 。

## 开始训练

``train_textual_inversion.py`` 以下是一个命令行示例（DreamBooth方法）。

```
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py 
    --dataset_config=<在数据准备中创建的.toml 文件> 
    --output_dir=<训练模型的输出目录>  
    --output_name=<训练模型输出时的文件名 > 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=1600 
    --learning_rate=1e-6 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --token_string=mychar4 --init_word=cute --num_vectors_per_token=4
```

``--token_string`` 指定训练时的token字符串。请在训练时包含此字符串作为提示（例如，如果token_string为mychar4，则提示应包含“mychar4 1girl”等）。该字符串的部分将被替换为Textual Inversion的新token并进行训练。对于DreamBooth，class+identifier形式的数据集，将token_string设置为token字符串是最简单和最可靠的方法。

您可以使用``--debug_dataset``查看替换后的token id以确认提示中是否包含token字符串。请检查是否存在``49408``以后的token。

```
input ids: tensor([[49406, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```
tokenizer已经拥有的单词（通常单词）无法使用。

``--init_word``选项指定在初始化嵌入时使用的源标记字符串。最好选择与要学习的概念接近的标记字符串。不能指定超过两个标记的字符串。

``--num_vectors_per_toke``选项指定在此训练中要使用多少个标记。使用更多标记会增加表达能力，但会消耗更多标记。例如，如果num_vectors_per_token=8，则指定的标记字符串将消耗8个标记（在一般提示的77个标记限制中）。

这些是Textual Inversion的主要选项。接下来与其他训练脚本相同。

`num_cpu_threads_per_process` 通常情况下，应将``num_cpu_threads_per_process``设置为1。

`pretrained_model_name_or_path` 指定用于追加训练的原始模型。可指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers本地磁盘上的模型目录、Diffusers模型ID（例如"stabilityai/stable-diffusion-2"）。

`output_dir` 指定保存训练后模型的文件夹。`output_name`指定模型文件名，不包括扩展名。使用`save_model_as`指定保存为safetensors格式。

`dataset_config` 指定`.toml`文件。在文件中，将批次大小指定为`1`，以降低内存消耗。

将训练步骤`max_train_steps`设置为10000。在此处，将学习率`learning_rate`设置为5e-6。

为了省去内存，使用`mixed_precision="fp16"`（在RTX30系列及以上，也可以指定`bf16`。请将其与在环境整备时指定的accelerate设置相匹配）。同时指定`gradient_checkpointing`。

为了使用内存消耗较少的8位AdamW优化器（用于将模型最优化以适应训练数据），需要指定`optimizer_type="AdamW8bit"`。

`xformers`，使用xformers的CrossAttention。如果尚未安装xformers或出现错误（取决于环境，例如mixed_precision="no"等），则可以指定`mem_eff_attn`选项，以使用省内存版CrossAttention（速度较慢）。

如果有足够的内存，请编辑.toml文件并将批次大小增加到大约8（这可能会提高速度和精度）。

### 常用选项

请参考以下文档以了解选项：

- Stable Diffusion 2.x或其派生模型
- 训练要求clip skip大于或等于2的模型
- 使用超过75个token的caption进行训练

### 关于Textual Inversion的batch size

相比于对整个模型进行学习的DreamBooth或fine tuning，Textual Inversion使用的内存更少，因此可以使用更大的batch size。

# Textual Inversion的其他主要选项

有关所有选项的详细信息，请参阅其他文档。

* `--weights`
  * 在训练之前加载预训练的embeddings，并从中继续训练。
* `--use_object_template`
  * 不使用标题，而是使用默认的物体模板字符串（例如，`a photo of a {}`）进行训练。这与官方实现相同，标题将被忽略。。
* `--use_style_template`
  * 不使用标题，而是使用默认的风格模板字符串进行训练（例如，`a painting in the style of {}`）。这与官方实现相同，标题将被忽略。

## 生成图像的脚本

在gen_img_diffusers.py脚本中，使用``--textual_inversion_embeddings``选项指定训练的embeddings文件（可以使用多个）。如果在提示中使用embeddings文件的文件名（不包括扩展名），则将应用该embeddings。

