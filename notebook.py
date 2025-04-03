import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import GPT2LMHeadModel

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    for k, v in sd_hf.items():
        print(k, v.shape)
    return GPT2LMHeadModel, k, model_hf, sd_hf, v


@app.cell
def _(sd_hf):
    sd_hf["transformer.wpe.weight"].view(-1)[:20]
    return


@app.cell
def _(sd_hf):
    import matplotlib.pyplot as plt

    plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
    return (plt,)


@app.cell
def _(plt, sd_hf):
    plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
    plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
    plt.plot(sd_hf["transformer.wpe.weight"][:, 250])
    return


@app.cell
def _(plt, sd_hf):
    plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300, :300], cmap="gray")
    return


@app.cell
def _():
    from transformers import pipeline, set_seed

    generator = pipeline("text-generation", model="gpt2")
    set_seed(1337)
    generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    return generator, pipeline, set_seed


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
