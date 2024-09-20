---
title: Building an Image Scraper with Python
date: '2024-07-11'
author: John Henry
tags:
  - Placeholder
  - Example
  - Lorem ipsum
  - JavaScript
  - TypeScript

draft: false
---
Scraping images from websites can be useful for a variety of tasks, including image classification and training machine learning models. In this post, we will learn how to create an image scraper using Python!

<!--more-->


This post is actually part 1 of 2 of a classification blog post. In the next blog post we will take our scraped images and use them to classify objects! 

All the code can be found [here](https://github.com/jhmejia/google-image-scraper/) if you want to get right into it.


## What is an Image Scraper?

An image scraper is a program that automatically downloads images from a website. It falls under the broader category of "web scraping," which involves extracting data from websites.

To make an image scraper, there's essentially 5 steps:

1. Get all your prerequisites set up
2. Open a website using a web testing library
3. Loop through the website in a way that can find all the images in full size
4.  Store those URLs
5.  Download the images at the URLs locally!

Not too bad, right?

It can be pretty simple to make a simple image scraper. The hard part, depending on the website, is step 3. For example, Google images only shows you image thumbnails unless you click on them, which aren't full size. Some websites take a while to load as well. 

## Prerequisites
For this we will be using the `Selenium` and `Requests` libraries, so you will need to get those:

```sh
  pip install selenium
  pip install requests
  pip install shutil
```

Other popular web scraping libraries include BeautifulSoup, Scrapy, and lxml. Each of these has its strengths, and I recommend exploring them if you're interested in web scraping beyond images.

## Establishing a Connection


First, we need to fetch the webpage we’re scraping. If you’re targeting a specific website, you can simply hardcode its URL. But since we want to scrape images based on various search queries, we need to make our URL dynamic.

```py
# Build the Google Query.
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))
```

Here, `wd` is short for `webdriver.Chrome()`, part of selenium. When it executes, it opens a Chrome browser window and navigates to the search results.

> [!tip]
>
> If you're using Edge, you can still use `webdriver.Chrome()`, since Edge is chromium-based.
> If you are using Safari or Firefox, use `webdriver.Safari()` or `webdriver.Firefox()` respectively.

### Finding Image URLS

Next, we need to find the urls to all the images. We first find the images of all the thumbnails by filtering through the CSS selector img.Q4LuWd . What does this mean? This mean that it's a thumbnail. I found this by using inspect element on the images (CTRL + Shift + I) 


How do we identify this using selenium?
Easy! just use wd.find_elements.

In this case you can just `thumbnail_results = wd.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")`
Looping Through and Clicking
Next on our list of importance is clicking through all of the thumbnails and trying to click to find the "full size version" of the image. 

<img src="images in posts/image_attribute_html.png">


This is what it will look like when navigating through thumbnails in the browser

<img src="images in posts/9PpnEOX.gif">

## Downloading images
Now you have to download the images from the url you found. You just loop through each url you retrieved and attempt to download it. 

We’ll use the Requests library to send a GET request to each URL.

We will make sure the status code is correct before downloading (HTTP status code 200 means OK, everything is working correctly.)

From there, we can save the file locally using Python's File-writing capabilities.
```py
try:
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image_number = i + quantity * index
        with open(folder_path + str(image_number) + '.jpg', 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
            print("SUCCESS- Got image" + str(image_number) + " - saved to " + folder_path + str(image_number) + '.jpg')
    else:
        print("ERROR - Could not download {url} - {r.status_code}")


except Exception as e:
    print(f"ERROR - Could not download {url} - {e}")
```

# Putting it all together
At this point, we're all done! I added a driver function that calls the functions of finding URLs and downloads each of these urls.

I also added a main.py where you can easily change the amount of images you want, what your search terms are, etc. 

```py
def search_and_download()
# Create a folder name.
    target_folder = os.path.join(target_path, "_".join(title_term.lower().split(" ")))

    # Open Chrome
    with webdriver.Chrome() as wd:
        # Search for images URLs.
        res = fetch_image_urls(
            search_term,
            number_images,
            wd=wd,
            sleep_between_interactions=SLEEP_BETWEEN_INTERACTIONS,
        )

        # Download the images.
        if res is not None:
            for i, elem in enumerate(res):
                download_image(target_folder, elem, i, number_images, index)
        else:
            print(f"Failed to return links for term: {search_term}")
```

# Conclusion


That’s it for part 1! You’ve now built a basic image scraper that can collect images from Google based on any search term. In part 2, we’ll explore how to use these images for object classification. Stay tuned!

Resources: 

- [Original Code](https://github.com/jhmejia/google-image-scraper/)