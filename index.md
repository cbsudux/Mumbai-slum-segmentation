# Mumbai slum segmentation

(Stress on AI 4 social good)

Add an intro (gives reader quick idea on what you have done, go into detail after talking about slums and giving slum facts) 

More than one billion people live in slums around the world. In some developing countries, slum residents make up for more than half of the population and lack reliable sanitation services, clean water, electricity, other basic services. We wanted to help. 

![intro-pic](/assets/images/combined-intro.png)


## What are slums?

The United Nations Habitat program defines slums as informal settlements that lack one or more of the following five conditions: access to clean water, access to improved sanitation, sufficient living area that is not overcrowded, durable housing and secure tenure.

## Mumbai Slums

Mumbai is one of the most populous cities in India, and while it is one of the wealthiest cities in India, it is also home to some of the world’s biggest slums -- Dharavi, Mankhurd-Govandi belt, Kurla-Ghatkopar belt, Dindoshi and The Bhandup-Mulund slums. The number of slum-dwellers in Mumbai is estimated to be around 9 million, up from 6 million in 2001 that is, 62% of of Mumbai live in informal slums.

![dharavi-govandi](/assets/images/dh-govandi.png)

![kurla](/assets/images/kurla.jpg)

(add one crude infographic of mumbai with markers of each slum?)

The situation is pretty bad in Mumbai. These pictures only capture a fraction of ... 

We wanted to help. 


## What did we do?

Any intitative on slum rehabitiation and improvement relies heavily on **slum mapping** and **monitoring**. When we spoke to the relevant authorities, we found out that they mapped slums manually (human annotators), which takes a substantial amount of time. We realised we could automate this and came up with a deep learning approach to **segment and map individual slums from satellite imagery**, leveraging CNNs for instance segmentation. In addition, we also wrote code to **perform change detection and monitor slum change over time**, which is an important task and a very good *urban economy* indicator.

image - predicted mask + image

Change detection

## How did we go about it?

We curated a **dataset** containing 3-band (RGB) satellite imagery with 65 cm per pixel resolution
collected from Google Earth. Each image has a pixel size of 1280x720. The satellite imagery covers most of
Mumbai and we include images from 2002 to 2018, to analyze slum change. We used 513 images for training, and 97 images for testing.

For slum segmentation and mapping, we trained a Mask R-CNN on our custom dataset. Check our [github readme](https://github.com/cbsudux/Mumbai-slum-segmentation/tree/master/slums) for our training and testing approaches, and our [paper](https://arxiv.org/abs/1811.07896) for more details.  

image result
video result


For slum change detection, we took a pair of satellite images, representing the same location at different points of time. We predicted masks for both these images and then subtract the masks to obtain a percentage icrease/decrease.   

image result


## Contributors

- [Sudharshan Chandra Babu](http://github.com/cbsudux)
- [Shishira R Maiya](https://github.com/abhyantrika)

## How can you help?

Very good section.

## Acknowledgements

We would like to thank the Slum Rehabiliation Authority of Mumbai for their data.

## Citing

We published our work in the NeurIPS (NIPS) 2018 ML4D workshop. 

## Additional Resources


