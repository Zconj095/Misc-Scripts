using UnityEngine;
using UnityEngine.UI;

public class SeasonDisplayUI : MonoBehaviour
{
	public SeasonManagementSystem SeasonManagementSystem;
	public Text seasonText; // Assign a UI Text element in the inspector

	void Update()
	{
		seasonText.text = "Season: " + SeasonManagementSystem.currentSeason.ToString();
		// You can also add more information like climate change factor
	}
}
