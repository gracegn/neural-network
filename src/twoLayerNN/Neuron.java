package twoLayerNN;

import java.util.Arrays;

public class Neuron {
	
	double bias;
	double weighted_input;
	
	public Neuron (double bias) {
		this.bias = bias;
	}
	
	public double calculate_output (double[] inputs, double[] weights) {
		this.weighted_input = calculate_weighted_input(inputs, weights);
		return NeuralNetwork.activate(weighted_input);
	}
	
	public double calculate_weighted_input (double[] inputs, double[] weights) {
		double result = this.bias;
		for (int i = 0; i < inputs.length; i++) {
			result += inputs[i] * weights[i];
		}
		
		return result;
	}
	
	public void update_bias (double bias) {
		this.bias = bias;
	}
	
	public void inspect (double[] weights) {
		System.out.println("\t      NEURON: weights = " + Arrays.toString(weights));
		System.out.println("\t              bias = " + bias);
	}
}