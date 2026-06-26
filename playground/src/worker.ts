import { pipeline, env } from '@xenova/transformers';

// Skip local model checks since we are running in browser and want to fetch from HF hub
env.allowLocalModels = false;
env.useBrowserCache = true;

class PipelineSingleton {
  static task: 'feature-extraction' = 'feature-extraction';
  static model = 'Xenova/all-MiniLM-L6-v2';
  static instance: any = null;

  static async getInstance(progress_callback?: Function) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, { progress_callback });
    }
    return this.instance;
  }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
  const { id, type, texts } = event.data;

  try {
    if (type === 'load') {
      // Initialize the model
      await PipelineSingleton.getInstance((x: any) => {
        self.postMessage({ id, status: 'progress', data: x });
      });
      self.postMessage({ id, status: 'ready' });
    } else if (type === 'embed') {
      // Get the pipeline instance
      const extractor = await PipelineSingleton.getInstance();
      
      // Compute embeddings
      const output = await extractor(texts, {
        pooling: 'mean',
        normalize: true,
      });

      // Convert to array of Float32Arrays
      const embeddings: Float32Array[] = [];
      const data = output.data;
      const dim = output.dims[1]; // Should be 384 for all-MiniLM-L6-v2
      
      for (let i = 0; i < texts.length; i++) {
        embeddings.push(new Float32Array(data.buffer, data.byteOffset + i * dim * 4, dim));
      }

      // Send the results back to the main thread
      self.postMessage({ id, status: 'complete', embeddings });
    }
  } catch (err) {
    self.postMessage({ id, status: 'error', error: err instanceof Error ? err.message : String(err) });
  }
});
