public class ShadowCacheTracker
{
    private long currentCacheSize = 0;
    private const long maxCacheSize = 30L * 1024 * 1024 * 1024; // 30 GB in bytes

    public void AddToCache(long size)
    {
        currentCacheSize += size;
        CheckCacheLimit();
    }

    private void CheckCacheLimit()
    {
        if (currentCacheSize >= maxCacheSize)
        {
            ClearCache();
        }
    }

    private void ClearCache()
    {
        // Logic to clear shadow cache
        // This could involve resetting lightmaps, clearing custom cache data, etc.
        currentCacheSize = 0;
    }
}
