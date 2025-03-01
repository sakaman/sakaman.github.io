const version = '20250301092644';
const cacheName = `static::${version}`;

const buildContentBlob = () => {
  return ["/database/2023/07/22/MySQL%E7%9A%84%E9%94%81%E7%8E%B0%E8%B1%A1%E5%8F%8A%E5%8E%9F%E7%90%86/","/reading/2023/06/30/The-Illustrated-Transformer/","/reading/2023/06/30/How-does-Stable-Diffusion-work/","/reading/2023/06/28/%E9%98%85%E8%AF%BB%E8%AE%BA%E6%96%87%E7%9A%84%E6%96%B9%E6%B3%95/","/%E9%9A%8F%E7%AC%94/2023/06/27/%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1/","/application/2023/06/06/%E4%B8%80%E8%87%B4%E6%80%A7%E6%A0%A1%E9%AA%8C/","/skills/2023/06/06/Prompts-for-ChatGPT/","/distributed%20system/2023/06/05/CAP/","/database/2022/09/05/Mysql%E6%8F%92%E5%85%A5%E6%80%A7%E8%83%BD/","/other/2022/06/28/%E5%B7%A5%E5%85%B7%E8%B5%84%E6%BA%90/","/about/","/categories/","/","/manifest.json","/assets/search.json","/search/","/style.css","/assets/css/style.css","/sitemap.xml","/robots.txt","/feed.xml","/blog/page2/","", "/assets/default-offline-image.png", "/assets/scripts/fetch.js"
  ]
}

const updateStaticCache = () => {
  return caches.open(cacheName).then(cache => {
    return cache.addAll(buildContentBlob());
  });
};

const clearOldCache = () => {
  return caches.keys().then(keys => {
    // Remove caches whose name is no longer valid.
    return Promise.all(
      keys
        .filter(key => {
          return key !== cacheName;
        })
        .map(key => {
          console.log(`Service Worker: removing cache ${key}`);
          return caches.delete(key);
        })
    );
  });
};

self.addEventListener("install", event => {
  event.waitUntil(
    updateStaticCache().then(() => {
      console.log(`Service Worker: cache updated to version: ${cacheName}`);
    })
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(clearOldCache());
});

self.addEventListener("fetch", event => {
  let request = event.request;
  let url = new URL(request.url);

  // Only deal with requests from the same domain.
  if (url.origin !== location.origin) {
    return;
  }

  // Always fetch non-GET requests from the network.
  if (request.method !== "GET") {
    event.respondWith(fetch(request));
    return;
  }

  // Default url returned if page isn't cached
  let offlineAsset = "/offline/";

  if (request.url.match(/\.(jpe?g|png|gif|svg)$/)) {
    // If url requested is an image and isn't cached, return default offline image
    offlineAsset = "/assets/default-offline-image.png";
  }

  // For all urls request image from network, then fallback to cache, then fallback to offline page
  event.respondWith(
    fetch(request).catch(async () => {
      return (await caches.match(request)) || caches.match(offlineAsset);
    })
  );
  return;
});
