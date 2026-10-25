## Background:

About the description-lengths research project, at a singular learning theory workshop. Links and interesting aspects for SLT should be highlighted.

The talk is 7 minutes, focus on the main points. Ignore details. Do not fill the slides with texts and numbers (noone cares if gain is 3891 bits or ~4k bits, we can read plots).

This is a document to revise the previously built presentation. What I liked:

- The plots
- The kappa and lambda pills

What I did not like:

- Mainly the verbosity. Way too much for a 7 minute talk, slides way too cluttered to allow me to direct the audience’s attention, they will get distracted by reading all the additional info.

Genral rules for main slides:

- Eye-catching visuals center
- At most 2-3 important conclusions, no details, nothing that the audience tries to read and gets distracted by while I talk
- Bottom right: Relevant cited papers (take relevant ones from talk_prep_notes.md)
- Backup slides can be more cluttered and verbose.

## Slide 1:

- Catchy question: something like “When we probe a network for an algorithm, can we determine if the network implements it or if we manufacture it?”
- Not much else. It should pique the interest of the audience

## Slide 2: Title slide:

- Title slide from the presentation is okay for this. Leave some space for information about me (name, position [postdoc at Laboratory of Data Engineering at AREA Science Park], contact info, …)

## Slide 3: Causal Abstractions

- Informative graphic using IOI on GPT-2 as an example:
    - 2 sentences, one ABAB, one ABBA (to show it’s not just about memorizing positions) in 2 lines
    - 2 arrows to randomly colored hidden states in some layer
    - 2 arrows with an alignment map, color the first ~4 hidden states different than the remaining ones (dark red with light red for 1st sentence, dark green with light green for the 2nd, for example)
    - 2 arrows pointing to a single hidden state that has the 4 dark green ones from the counterfactual and the light red origional ones
    - 1 arrow with the inverse alignment map, pointing to new randomly colored hidden states
    - 1 arrow pointing to the counterfactual output
- Keep the graphic simple. It should show the audience how causal abstraction works, I will point to the alignment map and say that this is what is learned, and that it can manufacture this by memorizing features in untrained models

## Slide 4: Description lenghts

- Centerpiece: Gain function. Color the different terms differently. This slide can remain relatively similar to slide 3 of the previous presentation. Important: A linear term governed by the expressibility of the function (or rather realizability), and a logarithmic term governed by the complexity

## Slide 5: “    **Interchange accuracy accepts explanations that the code rejects**   “

- The plot from the original slide. Takeaways: We can manufacture alignments regarding IIA that get rejected by code lengths. More complex alignment learners pay more.

## Slide 6: The complexity ladder

- Again, keep the plot, it’s nice

## Slide 7: Fit or Occam

- Just the top row of the plot. I will say in words that
    - The fit term discrepancy disappears with a learned temperature
    - The manufactured vs proper alignment using the coupling map for both pays 0 fit term, the entire difference is regret

## Slide 8: Investigations in kappa vs lambda

- 3-4 plots:
    - kappa as a function of subspace dimension
    - the plot from “    **½ tr(F H⁻¹) gets the order of κ; the nominal λ is 20–400× off**   “
    - The plot from “    **GPT-2: the information equality fails; the Fisher is 2 % of the Hessian**   “
    - Maybe another plot that you think is relevant, but it’s not necessary
- I will spend the remaining time I have on this slide going a bit into investigations of plug-in vs bayes codes and local kappa vs lambda

## Slide 9: Thank you & goodbye

- Call to action (contact me if you want to discuss, have ideas, or want to yell at me because I made a mistake), leave space for website, email, …

## Backup slides
- Put content previously in the main body of the presentation into backup slides. Sort out a couple backup slides, we don't need a whole lot, but also don't worry about it as I won't go through them but people will be able to download them and have a look.
