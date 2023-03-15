# Blog

This repository is for my personal blog built with [Hugo](https://gohugo.io/).
Here are some useful links and instructions for a quick setup.

## Theme
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) :heart:
- [Installation](https://github.com/adityatelange/hugo-PaperMod/wiki/Installation)

I would suggest using the method 1 because I found it's hard to customize the theme using a git submodule.

## Quarto
- [Quarto](https://quarto.org/) :heart:
- [Quarto for Hugo](https://quarto.org/docs/output-formats/hugo.html)

Just follow the steps and you will be able to run your code while writing with a quarto notebook. 

## Math 
- [Math Typesetting in Hugo](https://mertbakir.gitlab.io/hugo/math-typesetting-in-hugo/)
- [Math Typesetting](https://adityatelange.github.io/hugo-PaperMod/posts/math-typesetting/)
- [Render LaTex with KaTex in Hugo Blog](https://hyperupcall.github.io/blog/posts/render-latex-with-katex-in-hugo-blog/)


## Giscus

- [giscus](https://giscus.app/) :heart: Follow the configuration from the official page.
- Copy the code below and paste it to `/layouts/partials/comments.html`.
```html
{{ if site.Params.comments.giscus }}
<script src="https://giscus.app/client.js"
        data-repo="{{site.Params.comments.giscus.repo}}"
        data-repo-id="{{site.Params.comments.giscus.repo_id}}"
        data-category="{{site.Params.comments.giscus.category}}"
        data-category-id="{{site.Params.comments.giscus.category_id}}"
        data-mapping="{{site.Params.comments.giscus.mapping}}"
        data-reactions-enabled="{{site.Params.comments.giscus.reactions_enabled}}"
        data-theme="{{site.Params.comments.giscus.theme}}"
        data-language="{{site.Params.comments.giscus.lang}}"
        crossorigin="anonymous"
        async>
</script>
<noscript>Please enable JavaScript to view the comments powered by giscus.</noscript>
{{ end }}
```
- Add parameters to `config.yml`. The values in the brackets are from the script you obtained in the first step.

```yml
comments: 
    giscus:
        repo: "[ENTER REPO HERE]"
        repo-id: "[ENTER REPO ID HERE]"
        category: "[ENTER CATEGORY NAME HERE]"
        category-id: "[ENTER CATEGORY ID HERE]"
        mapping: "pathname"
        strict: "0"
        reactions_enabled: "1"
        theme: "preferred_color_scheme"
        lang: "en"
```

Happy blogging!