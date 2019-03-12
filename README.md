# APS360Project
------ APS360Project ------

Detecting If a Photo Has Been Manipulated / Photoshopped (Team 18)

With photo-altering tools becoming readily available to the public, it is becoming harder to
distinguish between which images are real and which are fake. Unhonest people create altered
photos that go on to be seen (and ultimately believed) by internet users. Photo-manipulation
scandals can be dangerous, especially with the popularity of social media platforms. The issue
has been given the popular term “fake news”, and Photoshop is among the biggest contributors.
Some recent false image scandals range from forged political campaign material, to fake
UFO sightings.

The team will be using the PS-Battles dataset that has been collected and cleaned by
members of a subreddit group. The dataset contains 11,142 original images, and 91,886
derivatives of the same images from the photoshop-battles subreddit, all of which are labelled.
The GitHub script is provided to download the dataset on a Mac computer. Since this is a binary
classification problem, we will separate the images into their two respective subfolders and use
the PyTorch Data Loader. There are multiple derivatives for each image, therefore it is important
to ensure the train, validation, and test sets don’t contain derivatives of the same image, as this
will cause overfitting. All the images vary in size and dimension, therefore they will all be
downsized to a certain size, but will not be cropped. Downsizing is also beneficial in terms of
computation, because otherwise large images could take unrealistic amounts of time to train.

