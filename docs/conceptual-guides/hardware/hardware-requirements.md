# Hardware Requirements

The following table is a simple reference for minimum VRAM requirements of particular LMs during inference, and NVIDIA (data center) hardware that meet those requirements.

> [!NOTE]
> the table assumes 16-bit (half) precision. as such, models quantized to lower bit representations may be capable of using less powerful devices than what is listed.

> [!NOTE]
> finetuning models will require different hardware, and is subject to data considerations

<table>
    <tr>
        <th>Model</th>
        <th>Hardware</th>
    </tr>
    <tr>
        <td>Llama 3.1 8B</td>
        <td>L4 or better</td>
    </tr>
    <tr>
        <td>Llama 3.1 70B</td>
        <td>multiple A100s or better</td>
    </tr>
    <tr>
        <td>Llama 3.2 1B</td>
        <td>T4 or better</td>
    </tr>
    <tr>
        <td>Llama 3.2 3B</td>
        <td>T4 or better</td>
    </tr>
    <tr>
        <td>Llama 3.2 11B Vision</td>
        <td>A10G or better</td>
    </tr>
    <tr>
        <td>Llama 3.2 90B Vision</td>
        <td>multiple A100s or better</td>
    </tr>
    <tr>
        <td>CodeLlama 7B</td>
        <td>A10G or better</td>
    </tr>
    <tr>
        <td>CodeLlama 13B</td>
        <td>A10G or better</td>
    </tr>
    <tr>
        <td>CodeLlama 34B</td>
        <td>L40S or better</td>
    </tr>
    <tr>
        <td>CodeLlama 70B</td>
        <td>A100 or better</td>
    </tr>
</table>

> [!TIP]
> 
Read [Understanding VRAM and how Much Your LLM Needs](https://blog.runpod.io/understanding-vram-and-how-much-your-llm-needs/) for more on VRAM requirements:
