import os
from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary
# https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common


app = Flask(__name__)

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 初始化推理管道
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn',
    model_revision='v2.0.4',
    vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', vad_model_revision="v2.0.4",
    punc_model='iic/punc_ct-transformer_cn-en-common-vocab471067-large', punc_model_revision="v2.0.4",
    spk_model="cam++", spk_model_revision="v2.0.2",
)


@app.route('/asr', methods=['POST'])
def asr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 保存上传的文件
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)

    # 进行推理
    rec_result = inference_pipeline(file_path, batch_size_s=300)

    # 删除临时文件
    os.remove(file_path)

    # 返回结果
    return jsonify(rec_result[0]["sentence_info"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)