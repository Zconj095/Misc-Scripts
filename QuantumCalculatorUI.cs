using UnityEngine;
using UnityEngine.UI;
using System;

public class QuantumCalculatorUI : MonoBehaviour
{
	public InputField numberOfParticlesInput;
	public Dropdown particleTypeDropdown;
	public QuantumStateCalculator stateCalculator;
	public Text resultsText;

	public void OnSubmit()
	{
		try
		{
			int numberOfParticles = int.Parse(numberOfParticlesInput.text);
			string particleType = particleTypeDropdown.options[particleTypeDropdown.value].text;

			// Validate inputs
			if (numberOfParticles <= 0)
			{
				Debug.LogError("Number of particles must be greater than 0");
				return;
			}

			// Call the QuantumStateCalculator with validated inputs
			QuantumState[] states = stateCalculator.CalculateQuantumStates(numberOfParticles, particleType);

			// Display the results of the calculation
			resultsText.text = "Quantum states:\n";
			foreach (QuantumState state in states)
			{
				resultsText.text += state.ToString() + "\n";
			}
		}
		catch (FormatException)
		{
			Debug.LogError("Invalid format for number of particles");
		}
	}
}
