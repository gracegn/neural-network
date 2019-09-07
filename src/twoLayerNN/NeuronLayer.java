package twoLayerNN;

import java.util.Arrays;

public class NeuronLayer {
	
	int num_neurons;
	int num_inputs;
	double[][] weights;
	Neuron[] neurons;
	
	public NeuronLayer (int num_neurons, double[] weights_arr, double bias) {
		this.num_neurons = num_neurons;
		this.num_inputs = weights_arr.length/this.num_neurons;
		this.weights = new double[this.num_neurons][];
		
		// converting weights array into a 2d array using indices w^l_jk format
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = Arrays.copyOfRange(weights_arr, i*num_inputs, (i+1)*num_inputs);
		}
		
		this.neurons = new Neuron[num_neurons];
		for (int i = 0; i < this.num_neurons; i++) {
			this.neurons[i] = new Neuron(bias);
		}
	}
	
	public double[] feedforward (double[] inputs) {
		double[] outputs = new double[num_neurons];
		
		for (int i = 0; i < num_neurons; i++) {
			outputs[i] = this.neurons[i].calculate_output(inputs, this.weights[i]);
		}
		return outputs;
	}
	
	// we take the product of weight matrix and deltas
	public double[] get_layer_deltas (double[] deltas, double[] outputs) {
		double[] products = new double[this.num_inputs];
		for (int i = 0; i < products.length; i++) {
			products[i] = 0;
			for (int j = 0; j < deltas.length; j++) {
				products[i] += this.weights[j][i] * deltas[j];
			}
			products[i] *= NeuralNetwork.activate_derivative(outputs[i]);
		}
		
		return products;
	}
	
	public double[][] get_gradients_w (double[] activations, double[] deltas) {
		double[][] gradients = new double[this.num_neurons][this.num_inputs];
		for (int j = 0; j < gradients.length; j++) {
			for (int k = 0; k < gradients[0].length; k++) {
				gradients[j][k] = activations[k]*deltas[j];
			}
		}
		
		return gradients;
	}
	
	public double[] get_gradients_b (double[] deltas) {
		return deltas;
	}
	
	public void update_network_layer (double[][] gradients_w, double[] gradients_b, double LEARNING_RATE) {
		for (int i = 0; i < gradients_w.length; i++)
			for (int j = 0; j < gradients_w[0].length; j++) {
				weights[i][j] -= LEARNING_RATE * gradients_w[i][j];
			}
		
		for (int i = 0; i < this.num_neurons; i++) {
			neurons[i].update_bias(gradients_b[i]);
		}
	}
	
	public void inspect() {
		System.out.println("\tLAYER");
		for (int i = 0; i < num_neurons; i++) {
			this.neurons[i].inspect(this.weights[i]);
		}
	}
}