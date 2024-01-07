using UnityEngine;

public class ShadowManager : MonoBehaviour
{
    private ShadowCacheTracker cacheTracker = new ShadowCacheTracker();

    void UpdateShadowCache(long estimatedShadowSize)
    {
        cacheTracker.AddToCache(estimatedShadowSize);
    }
}
