const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
 
async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
 
        const classes = ['Cancer', 'Non-cancer'];
 
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        
        const normalizedScore = score[0];
        const confidenceScore = normalizedScore * 100;
        
        const label = normalizedScore > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = label === 'Cancer' ? 'Segera periksa ke dokter!' : 
        'Penyakit kanker tidak terdeteksi.';

        return { confidenceScore, label, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`)
    }
}
 
module.exports = predictClassification;