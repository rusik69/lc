package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          200,
			Title:       "HTML Fundamentals",
			Description: "Learn the foundation of web development with HTML: structure, semantics, and best practices.",
			Order:       0,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to HTML",
					Content: `HTML (HyperText Markup Language) is the foundational markup language of the World Wide Web. Created in 1991 by Tim Berners-Lee, HTML provides the structural skeleton for every web page you visit, defining how content is organized, displayed, and understood by browsers, search engines, and assistive technologies. Understanding HTML deeply is essential for web development, as it forms the base layer that CSS styles and JavaScript enhances.

**What HTML Really Is:**

**Markup Language, Not Programming Language:**
HTML is a markup language, meaning it describes the structure and meaning of content rather than executing logic or performing calculations. It uses tags (markup) to label different parts of content, telling browsers how to interpret and display that content.

**Key Characteristics:**
- **Declarative**: Describes what content is, not how to process it
- **Hierarchical**: Elements nest inside other elements (tree structure)
- **Semantic**: Elements convey meaning about content
- **Standardized**: Defined by W3C (World Wide Web Consortium) and WHATWG

**How HTML Works:**
1. **Author writes HTML**: Developer creates HTML document
2. **Browser parses HTML**: Browser reads and interprets HTML
3. **DOM created**: Browser creates Document Object Model (DOM)
4. **Page rendered**: Browser displays page based on DOM
5. **User interacts**: User interacts with rendered page

**Historical Context:**

**The Birth of the Web (1991):**
- **Tim Berners-Lee**: Created HTML while working at CERN
- **Original Purpose**: Share scientific documents between researchers
- **First Web Page**: info.cern.ch (still accessible!)
- **Simple Design**: Basic tags for headings, paragraphs, and links
- **Evolution from SGML**: Based on SGML (Standard Generalized Markup Language)

**HTML Evolution Timeline:**

**HTML 1.0 (1991):**
- First version, never officially standardized
- Very basic: headings, paragraphs, links, lists
- No images, tables, or forms

**HTML 2.0 (1995):**
- First official standard (RFC 1866)
- Added forms, images, tables
- Established basic structure still used today

**HTML 3.2 (1997):**
- Added tables, applets, text flow
- More presentational elements
- W3C took over standardization

**HTML 4.01 (1999):**
- Major update, last version before HTML5
- Deprecated many presentational elements (moved to CSS)
- Strict, Transitional, and Frameset variants
- Still widely used today (legacy sites)

**XHTML 1.0 (2000):**
- XML-based version of HTML
- Stricter syntax requirements
- All tags must be closed, attributes quoted
- Less forgiving than HTML

**HTML5 (2014 - Current):**
- Major overhaul, current standard
- Introduced semantic elements (header, nav, article, etc.)
- New input types, audio/video elements
- Better form validation
- Improved accessibility features
- Living Standard: Continuously updated by WHATWG

**HTML5.1, HTML5.2, HTML5.3:**
- Incremental updates to HTML5
- New features and improvements
- Better accessibility, new APIs

**Why HTML Matters:**

**1. Foundation of the Web:**
- **Universal**: Every web page is built on HTML
- **Browser Support**: All browsers understand HTML
- **Essential Skill**: Required for any web developer
- **No Alternative**: There's no web without HTML
- **Real-world**: Over 1.9 billion websites use HTML

**2. Accessibility (Critical):**

**What is Web Accessibility:**
Making websites usable by people with disabilities, including:
- **Visual Impairments**: Blindness, low vision, color blindness
- **Hearing Impairments**: Deafness, hard of hearing
- **Motor Impairments**: Limited dexterity, paralysis
- **Cognitive Impairments**: Learning disabilities, attention disorders

**How HTML Enables Accessibility:**
- **Semantic HTML**: Screen readers understand structure
- **Alt Text**: Images described for visually impaired
- **Proper Headings**: Navigation for screen reader users
- **Form Labels**: Associates labels with form inputs
- **ARIA Attributes**: Additional accessibility information

**Legal Requirements:**
- **ADA (Americans with Disabilities Act)**: US law requires accessibility
- **WCAG (Web Content Accessibility Guidelines)**: International standard
- **Section 508**: US federal accessibility requirements
- **EN 301 549**: European accessibility standard
- **Real-world**: Companies face lawsuits for inaccessible websites

**3. SEO (Search Engine Optimization):**

**How Search Engines Use HTML:**
- **Crawl**: Search engines read HTML to understand content
- **Index**: HTML structure helps categorize content
- **Rank**: Semantic HTML improves search rankings
- **Display**: HTML meta tags appear in search results

**HTML Elements That Help SEO:**
- **Title Tag**: Appears in search results, critical for SEO
- **Meta Description**: Summary in search results
- **Heading Hierarchy**: H1-H6 help structure content
- **Semantic Elements**: Help search engines understand content
- **Alt Text**: Images indexed by search engines
- **Schema Markup**: Structured data for rich snippets

**Real-world Impact:**
- Proper HTML structure can improve search rankings significantly
- Semantic HTML helps search engines understand content better
- Good SEO drives organic traffic and business growth

**4. Maintainability:**

**Well-Structured HTML Benefits:**
- **Easier to Understand**: Clear structure is self-documenting
- **Easier to Modify**: Changes are straightforward
- **Easier to Style**: CSS targets clear structure
- **Easier to Script**: JavaScript manipulates clear DOM
- **Team Collaboration**: Others can understand your code

**Bad HTML Problems:**
- **Div Soup**: Overuse of divs makes structure unclear
- **Nested Tables**: Old layout technique, hard to maintain
- **Inline Styles**: Mixing presentation with structure
- **Poor Semantics**: Hard to understand content meaning

**5. Performance:**

**How HTML Affects Performance:**
- **Page Size**: Clean HTML is smaller, loads faster
- **Parsing Speed**: Well-formed HTML parses faster
- **Render Performance**: Proper structure renders efficiently
- **SEO Impact**: Faster pages rank better

**Performance Best Practices:**
- **Minimize HTML**: Remove unnecessary whitespace and comments
- **Proper Structure**: Helps browser optimize rendering
- **Lazy Loading**: Defer non-critical content
- **Semantic HTML**: Enables browser optimizations

**6. Cross-Browser Compatibility:**

**HTML Standards Ensure Compatibility:**
- **W3C Standards**: Ensure browsers interpret HTML consistently
- **Progressive Enhancement**: Start with HTML, enhance with CSS/JS
- **Graceful Degradation**: Works even if CSS/JS fails
- **Testing**: Valid HTML works across browsers

**Modern HTML Features:**

**HTML5 Semantic Elements:**
- **header**: Page or section header
- **nav**: Navigation links
- **main**: Main content area (one per page)
- **article**: Independent, reusable content
- **section**: Thematic grouping of content
- **aside**: Sidebar, complementary content
- **footer**: Page or section footer
- **figure/figcaption**: Images with captions
- **time**: Dates and times (machine-readable)

**Benefits of Semantic HTML:**
- **Accessibility**: Screen readers understand structure
- **SEO**: Search engines understand content hierarchy
- **Maintainability**: Code is self-documenting
- **Styling**: Easier to target with CSS
- **Future-Proof**: Works with new technologies

**HTML5 Form Enhancements:**
- **New Input Types**: email, url, tel, date, time, number, range, color
- **Built-in Validation**: Browser validates input automatically
- **Placeholder Text**: Hints for users
- **Required Fields**: Mark fields as required
- **Pattern Matching**: Regex validation
- **Better UX**: Native controls improve user experience

**HTML5 Media Elements:**
- **audio**: Native audio playback
- **video**: Native video playback
- **source**: Multiple format support
- **track**: Subtitles and captions
- **Benefits**: No plugins needed, better accessibility

**HTML5 APIs:**
- **Canvas**: 2D graphics and animations
- **SVG**: Vector graphics
- **Geolocation**: User location
- **Local Storage**: Client-side storage
- **Web Workers**: Background processing
- **WebSockets**: Real-time communication

**Best Practices (MDN Curriculum 2024-2025):**

**1. Use Semantic HTML:**
- Choose elements that convey meaning
- Use semantic elements (header, nav, article, etc.)
- Avoid div soup (overusing divs)
- **Example**: Use <nav> for navigation, not <div class="nav">
- **Why**: Improves accessibility, SEO, and maintainability

**2. Proper Document Structure:**
- Always include DOCTYPE
- Use proper heading hierarchy (H1 to H2 to H3)
- One H1 per page
- Logical content flow
- Use <main> element for main content (one per page)

**3. Accessibility First (Critical Gap Identified by MDN):**
- Always include alt text for images (descriptive, not "image" or "photo")
- Use proper form labels (associate with inputs using 'for' attribute or wrapping)
- Ensure keyboard navigation works (tab order, focus indicators)
- Test with screen readers (NVDA, JAWS, VoiceOver)
- Follow WCAG 2.1 Level AA guidelines (minimum standard)
- Use ARIA attributes when semantic HTML isn't sufficient
- Ensure color contrast meets WCAG standards (4.5:1 for normal text)
- **Legal Requirement**: ADA compliance required in many jurisdictions

**4. Performance and Responsive Design:**
- Mobile-first approach: Design for mobile, enhance for desktop
- Use responsive images (srcset, sizes attributes)
- Optimize images (WebP format, appropriate sizing)
- Minimize HTML size (remove unnecessary whitespace in production)
- Use lazy loading for below-the-fold content
- Consider Core Web Vitals (LCP, FID, CLS)

**5. Privacy Considerations:**
- Be transparent about data collection
- Use secure connections (HTTPS)
- Respect user privacy preferences
- Implement proper cookie consent
- Follow GDPR, CCPA, and other privacy regulations

**4. Valid HTML:**
- Validate your HTML
- Use W3C validator
- Fix errors and warnings
- Ensures cross-browser compatibility

**5. Separation of Concerns:**
- HTML for structure
- CSS for presentation
- JavaScript for behavior
- Keep concerns separated

**6. Performance:**
- Minimize HTML size (remove comments, whitespace in production)
- Use semantic HTML (enables browser optimizations)
- Lazy load non-critical content (loading="lazy" attribute)
- Optimize images and media (appropriate formats, sizes)
- Use modern image formats (WebP, AVIF) with fallbacks
- Preload critical resources (fonts, CSS, JavaScript)
- Minimize render-blocking resources

**7. Debugging and Problem-Solving Skills:**
- Use browser developer tools effectively (Chrome DevTools, Firefox DevTools)
- Validate HTML with W3C validator
- Test across multiple browsers (Chrome, Firefox, Safari, Edge)
- Use browser compatibility tools (Can I Use, MDN Browser Compatibility)
- Understand common HTML errors and how to fix them
- Learn to read and interpret error messages

**8. Soft Skills (MDN Curriculum Emphasis):**
- **Learning Mindset**: Continuous learning, staying updated with web standards
- **Collaboration**: Working effectively in teams, code reviews
- **Teamwork**: Contributing to open source, sharing knowledge
- **Feedback**: Giving and receiving constructive feedback
- **Research**: Finding solutions, reading documentation, asking questions
- **Planning**: Breaking down projects, estimating time, managing scope

**Common Mistakes:**

**1. Missing DOCTYPE:**
- **Problem**: Causes quirks mode, inconsistent rendering
- **Solution**: Always include <!DOCTYPE html>
- **Impact**: Pages render differently across browsers

**2. Improper Nesting:**
- **Problem**: Closing tags in wrong order
- **Example**: <p><strong>text</p></strong> (wrong)
- **Correct**: <p><strong>text</strong></p>
- **Impact**: Invalid HTML, rendering issues

**3. Forgetting Alt Text:**
- **Problem**: Images without alt attributes
- **Solution**: Always include alt text
- **Impact**: Hurts accessibility and SEO

**4. Using Presentational Elements:**
- **Problem**: Using deprecated elements (<font>, <center>, <b>)
- **Solution**: Use CSS instead, semantic alternatives
- **Impact**: Deprecated elements may not work in future

**5. Missing Closing Tags:**
- **Problem**: Some elements require closing tags
- **Solution**: Be explicit, don't rely on browser auto-closing
- **Impact**: Invalid HTML, potential rendering issues

**6. Div Soup:**
- **Problem**: Overusing divs instead of semantic elements
- **Solution**: Use semantic HTML5 elements
- **Impact**: Hard to maintain, poor accessibility

**Real-World Applications:**

**1. Web Development:**
- Foundation of all web pages
- Required for frontend development
- Works with CSS and JavaScript
- Essential for full-stack development

**2. Email Development:**
- HTML emails use HTML (with constraints)
- Table-based layouts often needed
- Inline styles required (limited CSS support)
- Different from web development

**3. Documentation:**
- Many documentation tools use HTML
- Markdown converts to HTML
- Static site generators output HTML
- Knowledge bases use HTML

**4. Web Scraping:**
- Understanding HTML structure helps scraping
- Parsing HTML to extract data
- CSS selectors for targeting elements
- XPath for complex queries

**5. Content Management:**
- CMSs output HTML
- Blog platforms use HTML
- Understanding HTML helps content creation
- WYSIWYG editors generate HTML

**Modern HTML Development:**

**HTML5 Living Standard:**
- Continuously updated by WHATWG
- New features added regularly
- Browser support varies
- Use feature detection

**Progressive Enhancement:**
- Start with HTML (works everywhere)
- Add CSS for styling
- Add JavaScript for interactivity
- Ensures basic functionality always works

**Web Components:**
- Custom elements
- Shadow DOM
- HTML templates
- Reusable components
- Future of web development

**Conclusion:**

HTML is the foundation of the web. Understanding HTML deeply is essential for building accessible, maintainable, and performant websites. Modern HTML5 provides powerful semantic elements, form enhancements, and APIs that enable rich web experiences while maintaining accessibility and SEO benefits.

Key principles:
- **Use semantic HTML**: Convey meaning, not just presentation
- **Accessibility first**: Make websites usable by everyone
- **Valid HTML**: Ensures cross-browser compatibility
- **Separation of concerns**: HTML for structure, CSS for presentation, JS for behavior
- **Progressive enhancement**: Start with HTML, enhance with CSS/JS

Remember: Good HTML is the foundation of good web development. Invest time in learning HTML properly, and it will pay dividends throughout your web development career. Whether you're building simple static sites or complex web applications, HTML is where it all starts.`,
					CodeExamples: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My First Web Page</title>
</head>
<body>
    <h1>Welcome to HTML</h1>
    <p>This is a paragraph.</p>
    <a href="https://example.com">Visit Example</a>
</body>
</html>`,
				},
				{
					Title: "HTML Elements and Tags",
					Content: `HTML elements are the building blocks of web pages. Each element has a start tag, content, and end tag.

**Common HTML Elements:**

**Headings:**
- h1 to h6 (h1 is most important, h6 is least)
- Used for document structure and hierarchy

**Paragraphs:**
- p element for text blocks
- Browser adds spacing automatically

**Links:**
- a element with href attribute
- Can link to external sites, internal pages, or sections

**Images:**
- img element with src and alt attributes
- alt text is crucial for accessibility

**Lists:**
- ul for unordered lists
- ol for ordered lists
- li for list items

**Semantic Elements (HTML5):**
- header, nav, main, article, section, aside, footer
- Improve document structure and accessibility`,
					CodeExamples: `<!-- Headings -->
<h1>Main Heading</h1>
<h2>Subheading</h2>
<h3>Sub-subheading</h3>

<!-- Paragraphs -->
<p>This is a paragraph of text.</p>
<p>Another paragraph with <strong>bold</strong> and <em>italic</em> text.</p>

<!-- Links -->
<a href="https://example.com">External Link</a>
<a href="/about.html">Internal Link</a>
<a href="#section1">Anchor Link</a>

<!-- Images -->
<img src="image.jpg" alt="Description of image" width="300" height="200">

<!-- Lists -->
<ul>
    <li>Unordered item 1</li>
    <li>Unordered item 2</li>
</ul>

<ol>
    <li>Ordered item 1</li>
    <li>Ordered item 2</li>
</ol>

<!-- Semantic HTML5 -->
<header>
    <h1>Site Title</h1>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
    </nav>
</header>
<main>
    <article>
        <h2>Article Title</h2>
        <p>Article content...</p>
    </article>
</main>
<footer>
    <p>&copy; 2024 My Site</p>
</footer>`,
				},
				{
					Title: "HTML Attributes",
					Content: `Attributes provide additional information about HTML elements. They are always specified in the start tag.

**Common Attributes:**

**Global Attributes (work on all elements):**
- id: Unique identifier
- class: CSS class selector
- style: Inline CSS (avoid when possible)
- title: Tooltip text
- data-*: Custom data attributes
- lang: Language specification

**Element-Specific Attributes:**
- href: Link destination (a element)
- src: Source URL (img, script, iframe)
- alt: Alternative text (img)
- type: Input type (input element)
- name: Form control name
- value: Default value
- required: Form validation
- disabled: Disable element
- readonly: Make read-only

**Best Practices:**
- Always use alt for images
- Use semantic class names
- Avoid inline styles
- Use data-* for custom attributes`,
					CodeExamples: `<!-- Global attributes -->
<div id="main-content" class="container" data-user-id="123" title="Main content area">
    Content here
</div>

<!-- Link attributes -->
<a href="https://example.com" target="_blank" rel="noopener noreferrer">
    External Link
</a>

<!-- Image attributes -->
<img src="photo.jpg" 
     alt="A beautiful sunset" 
     width="800" 
     height="600"
     loading="lazy">

<!-- Form input attributes -->
<input type="text" 
       name="username" 
       id="username" 
       placeholder="Enter username"
       required
       maxlength="20"
       autocomplete="username">

<!-- Button attributes -->
<button type="submit" disabled>Submit</button>
<button type="button" aria-label="Close dialog">×</button>

<!-- Data attributes for JavaScript -->
<div data-product-id="456" data-price="29.99">Product</div>`,
				},
				{
					Title: "HTML Forms and Tables",
					Content: `Forms and tables are essential HTML elements for data collection and display.

**HTML Forms:**
- form element: Container for form controls
- action: Where to send data
- method: GET or POST
- enctype: Encoding type for file uploads

**Form Controls:**
- input: Various input types (text, email, password, etc.)
- textarea: Multi-line text input
- select: Dropdown menu
- button: Submit or action button
- label: Associated with inputs for accessibility

**HTML Tables:**
- table: Container for tabular data
- thead, tbody, tfoot: Table sections
- tr: Table row
- th: Table header cell
- td: Table data cell
- colspan, rowspan: Cell spanning

**Best Practices:**
- Use semantic table structure
- Include table captions
- Use scope attribute for headers
- Avoid tables for layout (use CSS Grid/Flexbox)`,
					CodeExamples: `<!-- HTML Form -->
<form action="/submit" method="POST" enctype="multipart/form-data">
    <fieldset>
        <legend>User Information</legend>
        
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required>
        
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        
        <label for="bio">Bio:</label>
        <textarea id="bio" name="bio" rows="4" cols="50"></textarea>
        
        <label for="country">Country:</label>
        <select id="country" name="country">
            <option value="">Select...</option>
            <option value="us">United States</option>
            <option value="uk">United Kingdom</option>
        </select>
        
        <button type="submit">Submit</button>
    </fieldset>
</form>

<!-- HTML Table -->
<table>
    <caption>Monthly Sales Report</caption>
    <thead>
        <tr>
            <th scope="col">Month</th>
            <th scope="col">Sales</th>
            <th scope="col">Profit</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>January</td>
            <td>$10,000</td>
            <td>$2,000</td>
        </tr>
        <tr>
            <td>February</td>
            <td>$12,000</td>
            <td>$2,400</td>
        </tr>
    </tbody>
    <tfoot>
        <tr>
            <td>Total</td>
            <td>$22,000</td>
            <td>$4,400</td>
        </tr>
    </tfoot>
</table>

<!-- Table with colspan -->
<table>
    <tr>
        <th colspan="2">Name</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>John</td>
        <td>Doe</td>
        <td>30</td>
    </tr>
</table>`,
				},
				{
					Title: "HTML Multimedia and Meta Tags",
					Content: `Multimedia elements and meta tags enhance web pages with rich content and metadata.

**Multimedia Elements:**
- img: Images (use alt text)
- video: Video content
- audio: Audio content
- picture: Responsive images
- source: Multiple sources for media
- iframe: Embedded content

**Meta Tags:**
- charset: Character encoding
- viewport: Mobile responsiveness
- description: Page description (SEO)
- keywords: Page keywords (less important now)
- author: Page author
- og: Open Graph tags (social sharing)
- twitter: Twitter card tags

**Best Practices:**
- Always include alt text for images
- Use responsive images (srcset, sizes)
- Optimize media file sizes
- Include proper meta tags for SEO
- Use semantic HTML5 elements`,
					CodeExamples: `<!-- Images -->
<img src="photo.jpg" 
     alt="A beautiful sunset"
     srcset="photo-small.jpg 480w, photo-large.jpg 1200w"
     sizes="(max-width: 600px) 480px, 1200px"
     loading="lazy">

<!-- Responsive images with picture -->
<picture>
    <source media="(min-width: 800px)" srcset="large.jpg">
    <source media="(min-width: 400px)" srcset="medium.jpg">
    <img src="small.jpg" alt="Responsive image">
</picture>

<!-- Video -->
<video controls width="640" height="360">
    <source src="video.mp4" type="video/mp4">
    <source src="video.webm" type="video/webm">
    Your browser does not support the video tag.
</video>

<!-- Audio -->
<audio controls>
    <source src="audio.mp3" type="audio/mpeg">
    <source src="audio.ogg" type="audio/ogg">
    Your browser does not support the audio tag.
</audio>

<!-- Meta tags -->
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Learn HTML fundamentals">
    <meta name="keywords" content="HTML, web development, frontend">
    <meta name="author" content="John Doe">
    
    <!-- Open Graph (Facebook, LinkedIn) -->
    <meta property="og:title" content="HTML Fundamentals">
    <meta property="og:description" content="Learn HTML basics">
    <meta property="og:image" content="https://example.com/image.jpg">
    <meta property="og:url" content="https://example.com/page">
    
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="HTML Fundamentals">
    <meta name="twitter:description" content="Learn HTML basics">
    <meta name="twitter:image" content="https://example.com/image.jpg">
</head>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          201,
			Title:       "CSS Fundamentals",
			Description: "Master CSS styling: selectors, properties, values, and the cascade.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to CSS",
					Content: `CSS (Cascading Style Sheets) is used to style and layout web pages. It controls colors, fonts, spacing, positioning, and more.

**What is CSS?**
- Separates content (HTML) from presentation (CSS)
- Controls visual appearance of web pages
- Enables responsive design
- Allows consistent styling across pages

**CSS Syntax:**
- Selector: Targets HTML elements
- Property: What to style (color, font-size, etc.)
- Value: How to style it (red, 16px, etc.)

**Ways to Add CSS:**
1. Inline styles (style attribute)
2. Internal stylesheet (<style> tag)
3. External stylesheet (.css file) - Recommended

**CSS Selectors:**
- Element selector (p, div, h1)
- Class selector (.classname)
- ID selector (#idname)
- Attribute selector ([attribute])
- Pseudo-classes (:hover, :focus)
- Pseudo-elements (::before, ::after)`,
					CodeExamples: `/* External stylesheet (styles.css) */
/* Element selector */
p {
    color: blue;
    font-size: 16px;
}

/* Class selector */
.highlight {
    background-color: yellow;
}

/* ID selector */
#header {
    background-color: #333;
    color: white;
}

/* Multiple selectors */
h1, h2, h3 {
    font-family: Arial, sans-serif;
}

/* Descendant selector */
div p {
    margin: 10px;
}

/* Child selector */
ul > li {
    list-style: none;
}

/* Attribute selector */
input[type="text"] {
    border: 1px solid #ccc;
}

/* Pseudo-class */
a:hover {
    color: red;
    text-decoration: underline;
}

/* Pseudo-element */
p::first-line {
    font-weight: bold;
}`,
				},
				{
					Title: "CSS Properties and Values",
					Content: `CSS properties control specific aspects of element styling. Understanding common properties is essential.

**Text Properties:**
- color: Text color
- font-family: Font type
- font-size: Text size
- font-weight: Boldness (normal, bold, 100-900)
- text-align: Alignment (left, center, right, justify)
- text-decoration: Underline, overline, line-through
- line-height: Line spacing

**Box Model Properties:**
- width, height: Element dimensions
- margin: Space outside element
- padding: Space inside element
- border: Element border
- box-sizing: How dimensions are calculated

**Color Values:**
- Named colors (red, blue)
- Hex codes (#FF0000, #00FF00)
- RGB (rgb(255, 0, 0))
- RGBA (rgba(255, 0, 0, 0.5)) - with transparency
- HSL (hsl(0, 100%, 50%))

**Units:**
- px: Pixels (absolute)
- em: Relative to font-size
- rem: Relative to root font-size
- %: Percentage
- vw/vh: Viewport width/height`,
					CodeExamples: `/* Text styling */
.text-example {
    color: #333333;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 18px;
    font-weight: 400;
    text-align: center;
    text-decoration: none;
    line-height: 1.6;
    letter-spacing: 1px;
}

/* Box model */
.box {
    width: 300px;
    height: 200px;
    margin: 20px;
    padding: 15px;
    border: 2px solid #000;
    box-sizing: border-box; /* Includes padding/border in width */
}

/* Colors */
.color-examples {
    color: red;                    /* Named */
    background-color: #FF5733;    /* Hex */
    border-color: rgb(255, 87, 51); /* RGB */
    opacity: 0.8;                  /* Transparency */
}

/* Units */
.units-example {
    width: 50%;           /* Percentage */
    font-size: 1.2em;     /* Relative to parent */
    font-size: 1.2rem;   /* Relative to root */
    padding: 10px;        /* Pixels */
    margin: 2vw;          /* Viewport width */
    height: 50vh;         /* Viewport height */
}`,
				},
				{
					Title: "CSS Layout Basics",
					Content: `Understanding CSS layout is crucial for creating well-structured web pages.

**Display Property:**
- block: Takes full width, new line
- inline: Only takes needed width, no new line
- inline-block: Mix of both
- none: Hidden
- flex: Flexbox layout
- grid: Grid layout

**Position Property:**
- static: Default, normal flow
- relative: Relative to normal position
- absolute: Relative to nearest positioned ancestor
- fixed: Relative to viewport
- sticky: Switches between relative and fixed

**Float:**
- left, right: Float element
- none: No float (default)
- Used for wrapping text around images

**Common Layout Patterns:**
- Centering content
- Two-column layout
- Header, content, footer
- Navigation bars`,
					CodeExamples: `/* Display */
.block-element {
    display: block;
    width: 100%;
}

.inline-element {
    display: inline;
    margin: 0 10px;
}

.inline-block-element {
    display: inline-block;
    width: 200px;
    vertical-align: top;
}

/* Centering */
.center-box {
    width: 300px;
    margin: 0 auto; /* Centers block element */
}

.center-text {
    text-align: center;
}

/* Flexbox centering */
.flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

/* Position */
.relative {
    position: relative;
    top: 10px;
    left: 20px;
}

.absolute {
    position: absolute;
    top: 0;
    right: 0;
}

.fixed {
    position: fixed;
    bottom: 20px;
    right: 20px;
}

/* Two-column layout */
.two-column {
    display: flex;
}

.sidebar {
    width: 250px;
    flex-shrink: 0;
}

.main-content {
    flex: 1;
}`,
				},
				{
					Title: "CSS Specificity and Cascade",
					Content: `Understanding CSS specificity and cascade is crucial for predictable styling.

**CSS Specificity:**
- Determines which styles apply when rules conflict
- Calculated based on selector types
- Inline styles > IDs > Classes > Elements
- !important overrides everything (use sparingly)

**Specificity Calculation:**
- Inline style: 1000 points
- ID selector: 100 points
- Class/attribute/pseudo-class: 10 points
- Element/pseudo-element: 1 point
- Universal selector: 0 points

**Cascade Order:**
1. Importance (!important)
2. Specificity
3. Source order (last rule wins)

**Best Practices:**
- Avoid !important when possible
- Use classes instead of IDs for styling
- Keep specificity low
- Use BEM or similar naming conventions
- Organize CSS logically`,
					CodeExamples: `/* Specificity examples */
/* Specificity: 0,0,0,1 (1 point) */
p { color: blue; }

/* Specificity: 0,0,1,0 (10 points) */
.text { color: red; }

/* Specificity: 0,1,0,0 (100 points) */
#title { color: green; }

/* Specificity: 0,0,1,1 (11 points) */
p.text { color: purple; }

/* Specificity: 0,0,2,0 (20 points) */
.text.primary { color: orange; }

/* Inline style: 1000 points */
<p style="color: black;">Text</p>

/* !important overrides everything */
.text {
    color: red !important;
}

/* CSS Reset - low specificity */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* BEM naming convention */
.block { }
.block__element { }
.block--modifier { }
.block__element--modifier { }`,
				},
				{
					Title: "CSS Reset and Normalize",
					Content: `CSS reset and normalize ensure consistent styling across browsers.

**CSS Reset:**
- Removes default browser styles
- Provides clean slate
- Popular: Eric Meyer's Reset, Normalize.css
- Can be too aggressive

**Normalize.css:**
- Preserves useful defaults
- Fixes browser bugs
- More conservative approach
- Better for most projects

**Modern Approach:**
- Use CSS custom properties
- Set base styles
- Use modern reset (like modern-normalize)
- Consider CSS frameworks

**Common Reset Patterns:**
- Remove margins/padding
- Set box-sizing to border-box
- Normalize font rendering
- Remove list styles
- Set consistent line heights`,
					CodeExamples: `/* Basic CSS Reset */
*,
*::before,
*::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    -webkit-text-size-adjust: 100%;
    -moz-text-size-adjust: 100%;
    -ms-text-size-adjust: 100%;
    text-size-adjust: 100%;
}

body {
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

img,
picture,
video,
canvas,
svg {
    display: block;
    max-width: 100%;
}

input,
button,
textarea,
select {
    font: inherit;
}

p,
h1,
h2,
h3,
h4,
h5,
h6 {
    overflow-wrap: break-word;
}

/* Modern Normalize approach */
:root {
    --font-family: system-ui, -apple-system, sans-serif;
    --line-height: 1.5;
    --font-weight: 400;
}

body {
    font-family: var(--font-family);
    line-height: var(--line-height);
    font-weight: var(--font-weight);
}

/* Remove default list styles */
ul,
ol {
    list-style: none;
}

/* Remove default button styles */
button {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          202,
			Title:       "JavaScript Basics",
			Description: "Learn JavaScript fundamentals: variables, functions, control flow, and data types.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to JavaScript",
					Content: `JavaScript is a high-level, interpreted programming language that makes web pages interactive.

**What is JavaScript?**
- Adds interactivity to web pages
- Runs in the browser (client-side)
- Can also run on servers (Node.js)
- Dynamic, weakly typed language
- Object-oriented and functional programming support

**JavaScript in HTML:**
- Inline scripts (<script> tag)
- External scripts (src attribute)
- Event handlers (onclick, onload, etc.)

**JavaScript Versions:**
- ES5 (2009) - Widely supported
- ES6/ES2015+ - Modern features (let/const, arrow functions, classes)
- Modern browsers support ES6+

**Key Features:**
- Variables and data types
- Functions
- Control flow (if/else, loops)
- Objects and arrays
- DOM manipulation
- Event handling`,
					CodeExamples: `// Inline script
<script>
    console.log("Hello, World!");
</script>

// External script
<script src="script.js"></script>

// Variables (ES6)
let name = "John";
const age = 30;
var city = "New York"; // Avoid var, use let/const

// Data types
let number = 42;
let string = "Hello";
let boolean = true;
let nullValue = null;
let undefinedValue = undefined;
let object = { name: "John", age: 30 };
let array = [1, 2, 3];

// Functions
function greet(name) {
    return "Hello, " + name;
}

// Arrow function (ES6)
const greetArrow = (name) => {
    return 'Hello, ' + name;
};

// Shorter arrow function
const add = (a, b) => a + b;

// Console output
console.log("Debug message");
console.error("Error message");
console.warn("Warning message");`,
				},
				{
					Title: "Variables and Data Types",
					Content: `Understanding variables and data types is fundamental to JavaScript programming.

**Variable Declarations:**
- var: Function-scoped (avoid in modern code)
- let: Block-scoped, can be reassigned
- const: Block-scoped, cannot be reassigned (preferred for constants)

**Data Types:**
- Primitive types: string, number, boolean, null, undefined, symbol, bigint
- Object types: object, array, function, date, etc.

**Type Coercion:**
- JavaScript automatically converts types
- Can lead to unexpected results
- Use === for strict equality

**Template Literals:**
- Backticks for strings (use backtick character)
- Variable interpolation with ${}
- Multi-line strings

**Type Checking:**
- typeof operator
- instanceof operator
- Array.isArray()`,
					CodeExamples: `// Variable declarations
let count = 0;           // Can be reassigned
const PI = 3.14159;      // Cannot be reassigned
var oldWay = "avoid";    // Avoid var

// Data types
let str = "Hello";                    // String
let num = 42;                         // Number
let bool = true;                      // Boolean
let nothing = null;                  // Null (object type)
let notDefined = undefined;          // Undefined
let sym = Symbol("id");              // Symbol (ES6)
let bigNum = 9007199254740991n;      // BigInt (ES2020)

// Objects
let person = {
    name: "John",
    age: 30,
    city: "NYC"
};

// Arrays
let fruits = ["apple", "banana", "orange"];

// Type checking
console.log(typeof "hello");        // "string"
console.log(typeof 42);             // "number"
console.log(typeof true);           // "boolean"
console.log(typeof null);           // "object" (quirk)
console.log(Array.isArray([1,2]));  // true

// Template literals
const name = "John";
const greeting = 'Hello, ' + name + '!';
const multiLine = 
    'Line 1\n' +
    'Line 2\n' +
    'Line 3';

// Type coercion
console.log("5" + 3);    // "53" (string concatenation)
console.log("5" - 3);    // 2 (number subtraction)
console.log("5" == 5);   // true (loose equality)
console.log("5" === 5);  // false (strict equality)`,
				},
				{
					Title: "Functions and Control Flow",
					Content: `Functions are reusable blocks of code. Control flow determines the execution order.

**Function Types:**
- Function declarations
- Function expressions
- Arrow functions (ES6)
- Immediately Invoked Function Expressions (IIFE)

**Control Flow:**
- if/else statements
- switch statements
- Ternary operator
- Loops: for, while, do-while, for...in, for...of

**Function Parameters:**
- Default parameters (ES6)
- Rest parameters (...args)
- Arrow functions

**Scope:**
- Global scope
- Function scope
- Block scope (let/const)`,
					CodeExamples: `// Function declaration
function add(a, b) {
    return a + b;
}

// Function expression
const multiply = function(a, b) {
    return a * b;
};

// Arrow function
const divide = (a, b) => a / b;

// Default parameters
function greet(name = "Guest") {
    return 'Hello, ' + name;
}

// Rest parameters
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}

// If/else
let age = 18;
if (age >= 18) {
    console.log("Adult");
} else if (age >= 13) {
    console.log("Teen");
} else {
    console.log("Child");
}

// Ternary operator
const status = age >= 18 ? "Adult" : "Minor";

// Switch
let day = "Monday";
switch(day) {
    case "Monday":
        console.log("Start of week");
        break;
    case "Friday":
        console.log("Weekend!");
        break;
    default:
        console.log("Midweek");
}

// For loop
for (let i = 0; i < 5; i++) {
    console.log(i);
}

// For...of (arrays)
const fruits = ["apple", "banana", "orange"];
for (const fruit of fruits) {
    console.log(fruit);
}

// While loop
let count = 0;
while (count < 5) {
    console.log(count);
    count++;
}

// Array methods
fruits.forEach(fruit => console.log(fruit));
const doubled = [1, 2, 3].map(x => x * 2);
    const evens = [1, 2, 3, 4].filter(x => x % 2 === 0);`,
				},
				{
					Title: "Objects and Arrays",
					Content: `Objects and arrays are fundamental data structures in JavaScript.

**Objects:**
- Key-value pairs
- Properties and methods
- Object literal syntax
- Dot notation and bracket notation
- Object methods (Object.keys, Object.values, Object.entries)
- Destructuring objects

**Arrays:**
- Ordered lists of values
- Zero-indexed
- Dynamic length
- Array methods (map, filter, reduce, forEach, etc.)
- Array destructuring
- Spread operator

**Common Patterns:**
- Iterating over objects
- Transforming arrays
- Filtering and searching
- Combining arrays
- Nested structures`,
					CodeExamples: `// Objects
const person = {
    name: "John",
    age: 30,
    city: "NYC",
    greet() {
        return 'Hello, I\'m ' + this.name;
    }
};

// Access properties
person.name;           // "John"
person["age"];         // 30
person.greet();        // "Hello, I'm John"

// Add/modify properties
person.email = "john@example.com";
person.age = 31;

// Object methods
Object.keys(person);      // ["name", "age", "city", "greet", "email"]
Object.values(person);    // ["John", 31, "NYC", function, "john@example.com"]
Object.entries(person);   // [["name", "John"], ["age", 31], ...]

// Destructuring objects
const { name, age } = person;
const { name: personName, age: personAge } = person;

// Arrays
const fruits = ["apple", "banana", "orange"];
const numbers = [1, 2, 3, 4, 5];

// Array methods
fruits.forEach(fruit => console.log(fruit));
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
const sum = numbers.reduce((acc, n) => acc + n, 0);

// Array destructuring
const [first, second, ...rest] = fruits;
const [, , third] = fruits; // Skip first two

// Spread operator
const moreFruits = [...fruits, "grape", "mango"];
const combined = [...numbers, ...moreNumbers];

// Finding elements
const found = fruits.find(fruit => fruit.startsWith("a"));
const index = fruits.indexOf("banana");

// Array manipulation
fruits.push("grape");        // Add to end
fruits.pop();                 // Remove from end
fruits.unshift("kiwi");       // Add to beginning
fruits.shift();               // Remove from beginning
fruits.splice(1, 1);         // Remove at index`,
				},
				{
					Title: "Error Handling",
					Content: `Error handling prevents applications from crashing and provides better user experience.

**Error Types:**
- SyntaxError: Code syntax errors
- ReferenceError: Undefined variable
- TypeError: Wrong type operation
- RangeError: Value out of range
- Custom errors: User-defined errors

**Try-Catch-Finally:**
- try: Code that might throw errors
- catch: Handle errors
- finally: Always executes
- throw: Throw custom errors

**Error Handling Strategies:**
- Validate input
- Use try-catch for risky operations
- Provide meaningful error messages
- Log errors for debugging
- Handle errors gracefully

**Best Practices:**
- Don't catch errors you can't handle
- Provide user-friendly messages
- Log technical details
- Use error boundaries in React
- Validate data before processing`,
					CodeExamples: `// Basic try-catch
try {
    const result = riskyOperation();
    console.log(result);
} catch (error) {
    console.error("Error occurred:", error.message);
}

// Try-catch-finally
try {
    // Open file
    processFile();
} catch (error) {
    console.error("File processing failed:", error);
} finally {
    // Always close file
    closeFile();
}

// Catching specific errors
try {
    JSON.parse(invalidJson);
} catch (error) {
    if (error instanceof SyntaxError) {
        console.error("Invalid JSON syntax");
    } else {
        console.error("Unknown error:", error);
    }
}

// Throwing custom errors
function divide(a, b) {
    if (b === 0) {
        throw new Error("Division by zero is not allowed");
    }
    return a / b;
}

// Error handling in async code
async function fetchData() {
    try {
        const response = await fetch("/api/data");
        if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Fetch failed:", error);
        return null; // Return default value
    }
}

// Error handling with promises
fetch("/api/data")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error("Error:", error));

// Custom error class
class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

function validateEmail(email) {
    if (!email.includes("@")) {
        throw new ValidationError("Invalid email format");
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          203,
			Title:       "DOM Manipulation",
			Description: "Learn to interact with HTML elements using JavaScript: selecting, modifying, and event handling.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Understanding the DOM",
					Content: `The Document Object Model (DOM) is a programming interface for HTML documents. It represents the page structure as a tree of objects.

**What is the DOM?**
- Tree structure representing HTML
- Each HTML element is a node
- JavaScript can access and modify nodes
- Changes reflect immediately in the browser

**DOM Tree Structure:**
- Document (root)
- Elements (HTML tags)
- Text nodes
- Attributes
- Comments

**DOM Methods:**
- getElementById(): Get element by ID
- getElementsByClassName(): Get elements by class
- getElementsByTagName(): Get elements by tag
- querySelector(): Get first matching element
- querySelectorAll(): Get all matching elements

**DOM Properties:**
- innerHTML: HTML content
- textContent: Text content
- innerText: Visible text
- style: Inline styles
- className: CSS classes
- classList: Class list methods`,
					CodeExamples: `// Selecting elements
const elementById = document.getElementById("myId");
const elementsByClass = document.getElementsByClassName("myClass");
const elementsByTag = document.getElementsByTagName("div");
const firstMatch = document.querySelector(".myClass");
const allMatches = document.querySelectorAll("p");

// Modifying content
elementById.innerHTML = "<strong>New content</strong>";
elementById.textContent = "Plain text";
elementById.innerText = "Visible text";

// Modifying styles
elementById.style.color = "red";
elementById.style.backgroundColor = "blue";
elementById.style.fontSize = "20px";

// CSS classes
elementById.className = "new-class";
elementById.classList.add("active");
elementById.classList.remove("inactive");
elementById.classList.toggle("visible");
elementById.classList.contains("active"); // true/false

// Attributes
elementById.setAttribute("data-id", "123");
const value = elementById.getAttribute("data-id");
elementById.removeAttribute("data-id");

// Creating elements
const newDiv = document.createElement("div");
newDiv.textContent = "New element";
document.body.appendChild(newDiv);

// Removing elements
const parent = elementById.parentNode;
parent.removeChild(elementById);`,
				},
				{
					Title: "Event Handling",
					Content: `Events are actions that occur in the browser. JavaScript can listen for and respond to these events.

**Common Events:**
- click: Mouse click
- mouseover/mouseout: Mouse hover
- keydown/keyup: Keyboard input
- submit: Form submission
- load: Page/asset loaded
- change: Input value changed
- focus/blur: Input focus

**Event Handling Methods:**
- addEventListener(): Modern approach
- onclick attribute: Inline (avoid)
- onEvent properties: Less flexible

**Event Object:**
- Contains information about the event
- event.target: Element that triggered event
- event.preventDefault(): Prevent default behavior
- event.stopPropagation(): Stop event bubbling

**Event Delegation:**
- Attach listener to parent
- Use event.target to identify child
- Efficient for dynamic content`,
					CodeExamples: `// addEventListener (recommended)
const button = document.querySelector("#myButton");
button.addEventListener("click", function(event) {
    console.log("Button clicked!");
    console.log(event.target);
});

// Arrow function
button.addEventListener("click", (e) => {
    e.preventDefault();
    console.log("Prevented default");
});

// Multiple events
button.addEventListener("click", handleClick);
button.addEventListener("mouseover", handleHover);

function handleClick(e) {
    console.log("Clicked");
}

// Event object
document.addEventListener("keydown", (e) => {
    console.log("Key:", e.key);
    console.log("Code:", e.keyCode);
    if (e.key === "Enter") {
        console.log("Enter pressed");
    }
});

// Form events
const form = document.querySelector("#myForm");
form.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = form.querySelector("input");
    console.log("Value:", input.value);
});

// Event delegation
const list = document.querySelector("#myList");
list.addEventListener("click", (e) => {
    if (e.target.tagName === "LI") {
        console.log("List item clicked:", e.target.textContent);
    }
});

// Remove event listener
    button.removeEventListener("click", handleClick);`,
				},
				{
					Title: "Creating and Removing Elements",
					Content: `Dynamically creating and removing DOM elements enables interactive web applications.

**Creating Elements:**
- createElement(): Create new elements
- createTextNode(): Create text nodes
- cloneNode(): Clone existing elements
- innerHTML: Set HTML content (use carefully)

**Adding Elements:**
- appendChild(): Add to end
- insertBefore(): Insert before element
- prepend(): Add to beginning
- insertAdjacentHTML(): Insert HTML string

**Removing Elements:**
- removeChild(): Remove child element
- remove(): Remove element itself
- innerHTML = "": Clear all children

**Best Practices:**
- Use createElement for security
- Avoid innerHTML with user input
- Use DocumentFragment for performance
- Clean up event listeners when removing`,
					CodeExamples: `// Creating elements
const div = document.createElement("div");
div.className = "container";
div.textContent = "Hello World";

// Creating with attributes
const img = document.createElement("img");
img.src = "photo.jpg";
img.alt = "Photo";
img.width = 300;

// Creating text node
const text = document.createTextNode("Some text");
div.appendChild(text);

// Adding to DOM
document.body.appendChild(div);

// Inserting before
const newDiv = document.createElement("div");
const existingDiv = document.getElementById("existing");
existingDiv.parentNode.insertBefore(newDiv, existingDiv);

// Using insertAdjacentHTML
const container = document.getElementById("container");
container.insertAdjacentHTML("beforeend", "<p>New paragraph</p>");
// Positions: beforebegin, afterbegin, beforeend, afterend

// Cloning elements
const original = document.getElementById("original");
const clone = original.cloneNode(true); // true = deep clone
document.body.appendChild(clone);

// Removing elements
const element = document.getElementById("to-remove");
element.remove(); // Modern way

// Old way
const parent = element.parentNode;
parent.removeChild(element);

// Clearing all children
const container = document.getElementById("container");
container.innerHTML = "";

// Using DocumentFragment for performance
const fragment = document.createDocumentFragment();
for (let i = 0; i < 100; i++) {
    const li = document.createElement("li");
    li.textContent = 'Item ' + i;
    fragment.appendChild(li);
}
document.getElementById("list").appendChild(fragment);

// Creating complex structure
function createCard(title, content) {
    const card = document.createElement("div");
    card.className = "card";
    
    const cardTitle = document.createElement("h3");
    cardTitle.textContent = title;
    
    const cardContent = document.createElement("p");
    cardContent.textContent = content;
    
    card.appendChild(cardTitle);
    card.appendChild(cardContent);
    
    return card;
}

document.body.appendChild(createCard("Title", "Content"));`,
				},
				{
					Title: "DOM Traversal",
					Content: `DOM traversal allows navigation through the DOM tree structure.

**Parent Navigation:**
- parentNode: Direct parent
- parentElement: Parent element (null if not element)
- closest(): Nearest ancestor matching selector

**Child Navigation:**
- children: Element children only
- childNodes: All nodes (including text)
- firstChild/lastChild: First/last node
- firstElementChild/lastElementChild: First/last element

**Sibling Navigation:**
- nextSibling/previousSibling: Next/previous node
- nextElementSibling/previousElementSibling: Next/previous element

**Query Methods:**
- querySelector(): First matching element
- querySelectorAll(): All matching elements
- matches(): Check if element matches selector

**Best Practices:**
- Use element-specific properties when possible
- Prefer querySelector over traversal
- Cache references for performance
- Handle null checks`,
					CodeExamples: `// Parent navigation
const child = document.getElementById("child");
const parent = child.parentNode;
const grandparent = child.parentNode.parentNode;

// Using closest
const button = document.querySelector("button");
const card = button.closest(".card"); // Find nearest .card ancestor

// Child navigation
const container = document.getElementById("container");
const firstChild = container.firstElementChild;
const lastChild = container.lastElementChild;
const allChildren = Array.from(container.children);

// Iterating children
Array.from(container.children).forEach(child => {
    console.log(child);
});

// Sibling navigation
const element = document.getElementById("middle");
const next = element.nextElementSibling;
const previous = element.previousElementSibling;

// Find all siblings
function getAllSiblings(element) {
    const siblings = [];
    let sibling = element.parentNode.firstChild;
    while (sibling) {
        if (sibling.nodeType === 1 && sibling !== element) {
            siblings.push(sibling);
        }
        sibling = sibling.nextSibling;
    }
    return siblings;
}

// Traversing up to find specific ancestor
function findAncestor(element, selector) {
    let current = element;
    while (current) {
        if (current.matches(selector)) {
            return current;
        }
        current = current.parentElement;
    }
    return null;
}

// Finding all descendants
function findAllDescendants(element, selector) {
    return Array.from(element.querySelectorAll(selector));
}

// Walking the DOM tree
function walkDOM(node, callback) {
    callback(node);
    node = node.firstChild;
    while (node) {
        walkDOM(node, callback);
        node = node.nextSibling;
    }
}

walkDOM(document.body, (node) => {
    if (node.nodeType === 1) { // Element node
        console.log(node.tagName);
    }
});`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          204,
			Title:       "Responsive Design",
			Description: "Create websites that work on all devices: mobile-first approach, media queries, and flexible layouts.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Responsive Design",
					Content: `Responsive design ensures websites work well on all screen sizes and devices.

**What is Responsive Design?**
- Adapts to different screen sizes
- Single codebase for all devices
- Better user experience
- Improved SEO

**Key Concepts:**
- Fluid layouts (percentages, not fixed pixels)
- Flexible images and media
- CSS media queries
- Mobile-first approach

**Viewport Meta Tag:**
Essential for mobile devices:
<meta name="viewport" content="width=device-width, initial-scale=1.0">

**Breakpoints:**
Common device widths:
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

**Responsive Units:**
- %, em, rem: Relative units
- vw, vh: Viewport units
- min-width, max-width: Flexible sizing`,
					CodeExamples: `<!-- Viewport meta tag (required) -->
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<!-- Responsive images -->
<img src="image.jpg" 
     srcset="image-small.jpg 480w, image-large.jpg 1200w"
     sizes="(max-width: 600px) 480px, 1200px"
     alt="Responsive image">

<!-- Fluid container -->
.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Media queries */
/* Mobile first approach */
.responsive-box {
    width: 100%;
    padding: 10px;
}

/* Tablet */
@media (min-width: 768px) {
    .responsive-box {
        width: 50%;
        padding: 20px;
    }
}

/* Desktop */
@media (min-width: 1024px) {
    .responsive-box {
        width: 33.333%;
        padding: 30px;
    }
}

/* Max-width queries */
@media (max-width: 767px) {
    .hide-mobile {
        display: none;
    }
}

/* Orientation */
@media (orientation: landscape) {
    .landscape-layout {
        display: flex;
    }
}`,
				},
				{
					Title: "Media Queries",
					Content: `Media queries allow CSS to apply styles based on device characteristics.

**Media Query Syntax:**
@media media-type and (condition) { styles }

**Media Types:**
- all: All devices (default)
- screen: Computer screens, tablets, phones
- print: Printers
- speech: Screen readers

**Common Conditions:**
- min-width: Minimum width
- max-width: Maximum width
- orientation: portrait/landscape
- resolution: Screen resolution

**Logical Operators:**
- and: Combine conditions
- , (comma): OR condition
- not: Negate condition
- only: Hide from older browsers

**Best Practices:**
- Mobile-first (start small, scale up)
- Use relative units
- Test on real devices
- Consider touch targets (44px minimum)`,
					CodeExamples: `/* Mobile first (recommended) */
.base-styles {
    font-size: 14px;
    padding: 10px;
}

/* Tablet */
@media (min-width: 768px) {
    .base-styles {
        font-size: 16px;
        padding: 20px;
    }
}

/* Desktop */
@media (min-width: 1024px) {
    .base-styles {
        font-size: 18px;
        padding: 30px;
    }
}

/* Desktop large */
@media (min-width: 1440px) {
    .base-styles {
        font-size: 20px;
        padding: 40px;
    }
}

/* Multiple conditions */
@media (min-width: 768px) and (max-width: 1023px) {
    .tablet-only {
        display: block;
    }
}

/* OR condition */
@media (max-width: 767px), (orientation: portrait) {
    .mobile-portrait {
        display: block;
    }
}

/* Print styles */
@media print {
    .no-print {
        display: none;
    }
    
    body {
        color: black;
        background: white;
    }
}

/* High resolution screens */
@media (min-resolution: 192dpi) {
    .high-res-image {
        background-image: url("image@2x.jpg");
    }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    body {
        background: #000;
        color: #fff;
    }
}`,
				},
				{
					Title: "Mobile-First and Container Queries",
					Content: `Mobile-first design and container queries are modern responsive design approaches.

**Mobile-First Approach:**
- Start with mobile styles (base)
- Add styles for larger screens (min-width)
- Better performance on mobile
- Progressive enhancement

**Viewport Units:**
- vw: Viewport width (1vw = 1% of viewport width)
- vh: Viewport height (1vh = 1% of viewport height)
- vmin: Smaller of vw or vh
- vmax: Larger of vw or vh
- Useful for full-screen layouts

**Container Queries:**
- Query container size, not viewport
- @container rule (new feature)
- More flexible than media queries
- Better for component-based design

**Best Practices:**
- Always use mobile-first
- Test on real devices
- Use relative units
- Consider container queries for components`,
					CodeExamples: `/* Mobile-first approach */
/* Base styles (mobile) */
.container {
    padding: 10px;
    font-size: 14px;
}

/* Tablet and up */
@media (min-width: 768px) {
    .container {
        padding: 20px;
        font-size: 16px;
    }
}

/* Desktop and up */
@media (min-width: 1024px) {
    .container {
        padding: 30px;
        font-size: 18px;
    }
}

/* Viewport units */
.full-screen {
    width: 100vw;
    height: 100vh;
}

.half-viewport {
    width: 50vw;
    height: 50vh;
}

/* Responsive typography */
h1 {
    font-size: clamp(1.5rem, 4vw, 3rem);
}

/* Container queries (modern browsers) */
.card-container {
    container-type: inline-size;
}

@container (min-width: 400px) {
    .card {
        display: flex;
        flex-direction: row;
    }
}

@container (min-width: 600px) {
    .card {
        padding: 2rem;
    }
}

/* Fluid typography with viewport */
.responsive-text {
    font-size: calc(1rem + 1vw);
}

/* Aspect ratio with viewport */
.video-container {
    width: 100%;
    aspect-ratio: 16 / 9;
}

/* Full-height sections */
.section {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          205,
			Title:       "CSS Flexbox & Grid",
			Description: "Master modern CSS layout: Flexbox for one-dimensional layouts and Grid for two-dimensional layouts.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "CSS Flexbox",
					Content: `Flexbox is a one-dimensional layout method for arranging items in rows or columns.

**Flexbox Concepts:**
- Flex container (parent)
- Flex items (children)
- Main axis: Primary direction
- Cross axis: Perpendicular direction

**Container Properties:**
- display: flex or inline-flex
- flex-direction: row (default), column, row-reverse, column-reverse
- flex-wrap: nowrap (default), wrap, wrap-reverse
- justify-content: Main axis alignment
- align-items: Cross axis alignment
- align-content: Multi-line alignment
- gap: Space between items

**Item Properties:**
- flex-grow: Grow factor
- flex-shrink: Shrink factor
- flex-basis: Initial size
- flex: Shorthand (grow shrink basis)
- align-self: Override container alignment
- order: Visual order

**Common Patterns:**
- Centering content
- Navigation bars
- Card layouts
- Equal height columns`,
					CodeExamples: `/* Flex container */
.flex-container {
    display: flex;
    flex-direction: row;        /* or column */
    flex-wrap: wrap;            /* or nowrap */
    justify-content: center;    /* Main axis */
    align-items: center;         /* Cross axis */
    gap: 20px;
}

/* Centering */
.center-flex {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

/* Navigation bar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar .logo {
    margin-right: auto;
}

/* Card layout */
.card-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.card {
    flex: 1 1 300px;  /* grow shrink basis */
    min-width: 0;
}

/* Equal height columns */
.columns {
    display: flex;
}

.column {
    flex: 1;
}

/* Flex item properties */
.flex-item {
    flex-grow: 1;
    flex-shrink: 0;
    flex-basis: 200px;
    
    /* Shorthand */
    flex: 1 0 200px;
    
    align-self: flex-start;
    order: 2;
}

/* Responsive flex */
@media (max-width: 768px) {
    .flex-container {
        flex-direction: column;
    }
}`,
				},
				{
					Title: "CSS Grid",
					Content: `CSS Grid is a two-dimensional layout system for creating complex layouts.

**Grid Concepts:**
- Grid container (parent)
- Grid items (children)
- Grid lines: Dividing lines
- Grid tracks: Rows and columns
- Grid cells: Intersections
- Grid areas: Named regions

**Container Properties:**
- display: grid or inline-grid
- grid-template-columns: Column sizes
- grid-template-rows: Row sizes
- grid-template-areas: Named areas
- gap: Space between tracks
- justify-items: Horizontal alignment
- align-items: Vertical alignment
- place-items: Shorthand

**Track Sizing:**
- fr: Fractional unit
- auto: Content-based
- minmax(): Min/max sizing
- repeat(): Repeat pattern

**Item Properties:**
- grid-column: Column placement
- grid-row: Row placement
- grid-area: Area placement
- justify-self: Horizontal alignment
- align-self: Vertical alignment`,
					CodeExamples: `/* Basic grid */
.grid-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: auto;
    gap: 20px;
}

/* Grid with different column sizes */
.custom-grid {
    display: grid;
    grid-template-columns: 200px 1fr 200px;
    grid-template-rows: 100px auto 100px;
    gap: 20px;
}

/* Responsive grid */
.responsive-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

/* Grid areas */
.layout-grid {
    display: grid;
    grid-template-areas:
        "header header header"
        "sidebar main aside"
        "footer footer footer";
    grid-template-columns: 200px 1fr 200px;
    grid-template-rows: auto 1fr auto;
    gap: 20px;
}

.header { grid-area: header; }
.sidebar { grid-area: sidebar; }
.main { grid-area: main; }
.aside { grid-area: aside; }
.footer { grid-area: footer; }

/* Item placement */
.grid-item {
    grid-column: 1 / 3;      /* Start at 1, end at 3 */
    grid-row: 2 / 4;         /* Start at 2, end at 4 */
    
    /* Span */
    grid-column: span 2;
    grid-row: span 2;
    
    /* Named lines */
    grid-column: sidebar-start / main-end;
}

/* Alignment */
.grid-container {
    justify-items: center;    /* Horizontal */
    align-items: center;      /* Vertical */
    place-items: center;      /* Both */
}

.grid-item {
    justify-self: end;
    align-self: start;
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          206,
			Title:       "Forms and Validation",
			Description: "Create interactive forms with proper validation, accessibility, and user experience.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "HTML Forms",
					Content: `Forms collect user input and send it to a server for processing.

**Form Elements:**
- form: Container for form controls
- input: Text, email, password, number, etc.
- textarea: Multi-line text
- select: Dropdown menu
- button: Submit or action button
- label: Associated with inputs
- fieldset: Group related fields
- legend: Fieldset title

**Input Types:**
- text: Single-line text
- email: Email address
- password: Password (masked)
- number: Numeric input
- date: Date picker
- checkbox: Multiple selections
- radio: Single selection from group
- file: File upload
- submit: Submit button
- reset: Reset form

**Form Attributes:**
- action: Submission URL
- method: GET or POST
- name: Form identifier
- autocomplete: Browser autocomplete
- novalidate: Skip HTML5 validation`,
					CodeExamples: `<!-- Basic form -->
<form action="/submit" method="POST">
    <label for="username">Username:</label>
    <input type="text" id="username" name="username" required>
    
    <label for="email">Email:</label>
    <input type="email" id="email" name="email" required>
    
    <label for="password">Password:</label>
    <input type="password" id="password" name="password" required minlength="8">
    
    <label for="age">Age:</label>
    <input type="number" id="age" name="age" min="18" max="100">
    
    <label for="bio">Bio:</label>
    <textarea id="bio" name="bio" rows="4" cols="50"></textarea>
    
    <label for="country">Country:</label>
    <select id="country" name="country">
        <option value="">Select...</option>
        <option value="us">United States</option>
        <option value="uk">United Kingdom</option>
    </select>
    
    <fieldset>
        <legend>Gender</legend>
        <input type="radio" id="male" name="gender" value="male">
        <label for="male">Male</label>
        
        <input type="radio" id="female" name="gender" value="female">
        <label for="female">Female</label>
    </fieldset>
    
    <label>
        <input type="checkbox" name="terms" required>
        I agree to the terms
    </label>
    
    <button type="submit">Submit</button>
</form>`,
				},
				{
					Title: "Form Validation",
					Content: `Validation ensures users enter correct data before submission.

**HTML5 Validation:**
- required: Field must be filled
- min/max: Numeric/date limits
- minlength/maxlength: Text length
- pattern: Regex pattern
- type: Email, URL, etc.

**JavaScript Validation:**
- checkValidity(): Check if valid
- setCustomValidity(): Custom message
- validity object: Validation state
- Constraint Validation API

**Validation Strategies:**
- Client-side: Immediate feedback
- Server-side: Security (always required)
- Real-time: Validate on input
- On submit: Validate before sending

**Accessibility:**
- Associate labels with inputs
- Use aria-invalid for errors
- Provide error messages
- Use fieldset for groups`,
					CodeExamples: `<!-- HTML5 validation -->
<input type="email" 
       required 
       pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$"
       minlength="5"
       maxlength="50">

<!-- JavaScript validation -->
<form id="myForm">
    <input type="email" id="email" required>
    <span id="emailError" class="error"></span>
    <button type="submit">Submit</button>
</form>

<script>
const form = document.getElementById("myForm");
const email = document.getElementById("email");
const error = document.getElementById("emailError");

// Real-time validation
email.addEventListener("input", () => {
    if (email.validity.valid) {
        error.textContent = "";
        email.setAttribute("aria-invalid", "false");
    } else {
        showError();
    }
});

// Form submission
form.addEventListener("submit", (e) => {
    if (!form.checkValidity()) {
        e.preventDefault();
        showError();
    }
});

function showError() {
    if (email.validity.valueMissing) {
        error.textContent = "Email is required";
    } else if (email.validity.typeMismatch) {
        error.textContent = "Please enter a valid email";
    }
    email.setAttribute("aria-invalid", "true");
}

// Custom validation
email.addEventListener("input", () => {
    if (email.value.includes("test")) {
        email.setCustomValidity("Cannot use 'test' in email");
    } else {
        email.setCustomValidity("");
    }
});
</script>

/* CSS for validation states */
input:invalid {
    border-color: red;
}

input:valid {
    border-color: green;
}

.error {
    color: red;
    font-size: 0.875rem;
    display: block;
    margin-top: 5px;
}`,
				},
				{
					Title: "File Uploads and FormData API",
					Content: `File uploads and FormData enable sending files and complex form data.

**File Input:**
- input type="file": File selection
- accept: Allowed file types
- multiple: Allow multiple files
- Files property: Access selected files

**FormData API:**
- FormData object: Key-value pairs
- append(): Add fields
- get(): Get value
- getAll(): Get all values
- entries(): Iterate entries

**File Handling:**
- FileReader API: Read file contents
- File object: File metadata
- Blob: Binary large object
- URL.createObjectURL(): Create object URL

**Best Practices:**
- Validate file types and sizes
- Show upload progress
- Handle errors gracefully
- Use FormData for AJAX uploads`,
					CodeExamples: `<!-- File input -->
<input type="file" id="fileInput" accept="image/*" multiple>

<!-- File input with custom styling -->
<label for="fileInput" class="file-label">
    Choose Files
    <input type="file" id="fileInput" hidden>
</label>

<script>
// Accessing files
const fileInput = document.getElementById("fileInput");
fileInput.addEventListener("change", (e) => {
    const files = e.target.files;
    console.log('Selected ' + files.length + ' file(s)');
    
    Array.from(files).forEach(file => {
        console.log('Name: ' + file.name);
        console.log('Size: ' + file.size + ' bytes');
        console.log('Type: ' + file.type);
    });
});

// FormData for file upload
const formData = new FormData();
formData.append("username", "john");
formData.append("avatar", fileInput.files[0]);

// Upload with fetch
fetch("/upload", {
    method: "POST",
    body: formData
})
.then(response => response.json())
.then(data => console.log("Upload successful:", data))
.catch(error => console.error("Upload failed:", error));

// FileReader to read file contents
function readFile(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        console.log("File content:", e.target.result);
    };
    
    reader.onerror = (e) => {
        console.error("Error reading file:", e);
    };
    
    // Read as text
    reader.readAsText(file);
    
    // Or read as data URL (for images)
    // reader.readAsDataURL(file);
    
    // Or read as array buffer (for binary)
    // reader.readAsArrayBuffer(file);
}

// Preview image before upload
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.createElement("img");
        img.src = e.target.result;
        document.body.appendChild(img);
    };
    reader.readAsDataURL(file);
}

// Validate file
function validateFile(file) {
    const maxSize = 5 * 1024 * 1024; // 5MB
    const allowedTypes = ["image/jpeg", "image/png", "image/gif"];
    
    if (file.size > maxSize) {
        return { valid: false, error: "File too large" };
    }
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, error: "Invalid file type" };
    }
    
    return { valid: true };
}

// Upload with progress
function uploadWithProgress(file) {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);
    
    xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            console.log('Upload progress: ' + percentComplete + '%');
        }
    });
    
    xhr.addEventListener("load", () => {
        console.log("Upload complete");
    });
    
    xhr.open("POST", "/upload");
    xhr.send(formData);
}
</script>`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          207,
			Title:       "Accessibility Basics",
			Description: "Make websites accessible to all users: semantic HTML, ARIA attributes, and keyboard navigation.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Web Accessibility Fundamentals",
					Content: `Web accessibility ensures websites are usable by everyone, including people with disabilities.

**Why Accessibility Matters:**
- Legal requirement (ADA, WCAG)
- Better user experience for all
- Improved SEO
- Larger audience reach
- Ethical responsibility

**Types of Disabilities:**
- Visual: Blindness, low vision, color blindness
- Auditory: Deafness, hard of hearing
- Motor: Limited movement, tremors
- Cognitive: Learning disabilities, attention disorders

**WCAG Guidelines:**
- Perceivable: Information must be presentable
- Operable: Interface must be usable
- Understandable: Content must be clear
- Robust: Content must work with assistive tech

**Key Principles:**
- Semantic HTML
- Proper heading hierarchy
- Alt text for images
- Keyboard navigation
- Color contrast
- ARIA attributes`,
					CodeExamples: `<!-- Semantic HTML -->
<header>
    <nav aria-label="Main navigation">
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/about">About</a></li>
        </ul>
    </nav>
</header>

<main>
    <article>
        <h1>Article Title</h1>
        <p>Content...</p>
    </article>
</main>

<footer>
    <p>&copy; 2024</p>
</footer>

<!-- Proper heading hierarchy -->
<h1>Main Title</h1>
<h2>Section Title</h2>
<h3>Subsection Title</h3>

<!-- Alt text for images -->
<img src="photo.jpg" alt="A sunset over mountains">
<img src="decorative.jpg" alt=""> <!-- Decorative only -->

<!-- Form labels -->
<label for="email">Email Address</label>
<input type="email" id="email" name="email" required>

<!-- ARIA labels -->
<button aria-label="Close dialog">×</button>
<div role="button" tabindex="0" aria-label="Click to expand">Expand</div>

<!-- Skip links -->
<a href="#main-content" class="skip-link">Skip to main content</a>`,
				},
				{
					Title: "ARIA and Keyboard Navigation",
					Content: `ARIA (Accessible Rich Internet Applications) enhances accessibility for assistive technologies.

**ARIA Attributes:**
- aria-label: Accessible name
- aria-labelledby: Reference to label
- aria-describedby: Reference to description
- aria-hidden: Hide from screen readers
- aria-live: Dynamic content updates
- role: Element role

**Keyboard Navigation:**
- Tab: Navigate forward
- Shift+Tab: Navigate backward
- Enter/Space: Activate
- Arrow keys: Navigate options
- Escape: Close/dismiss

**Focus Management:**
- tabindex: Control focus order
- :focus styles: Visible focus indicators
- Focus trap: Keep focus in modal
- Focus restoration: Return focus after close

**Best Practices:**
- All interactive elements keyboard accessible
- Visible focus indicators
- Logical tab order
- Skip links for navigation
- No keyboard traps`,
					CodeExamples: `<!-- ARIA examples -->
<!-- Button with accessible name -->
<button aria-label="Add to cart">+</button>

<!-- Form with description -->
<label for="password">Password</label>
<input type="password" 
       id="password" 
       aria-describedby="password-help">
<span id="password-help">Must be at least 8 characters</span>

<!-- Live region for updates -->
<div aria-live="polite" aria-atomic="true">
    <span id="status">Status messages appear here</span>
</div>

<!-- Modal dialog -->
<div role="dialog" 
     aria-modal="true" 
     aria-labelledby="dialog-title">
    <h2 id="dialog-title">Confirm Action</h2>
    <button aria-label="Close dialog">×</button>
</div>

<!-- Keyboard navigation -->
<div class="menu" role="menubar">
    <button role="menuitem" tabindex="0">File</button>
    <button role="menuitem" tabindex="-1">Edit</button>
    <button role="menuitem" tabindex="-1">View</button>
</div>

/* Focus styles */
button:focus,
a:focus,
input:focus {
    outline: 2px solid blue;
    outline-offset: 2px;
}

.skip-link {
    position: absolute;
    left: -9999px;
}

.skip-link:focus {
    left: 0;
    top: 0;
    z-index: 9999;
}

/* JavaScript focus management */
function openModal() {
    const modal = document.getElementById("modal");
    const firstFocusable = modal.querySelector("button");
    modal.style.display = "block";
    firstFocusable.focus();
}

function closeModal() {
    const modal = document.getElementById("modal");
    const previousFocus = document.activeElement;
    modal.style.display = "none";
    previousFocus.focus();
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          208,
			Title:       "Browser APIs",
			Description: "Explore browser APIs: Local Storage, Fetch API, Geolocation, and more.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Storage APIs",
					Content: `Browser storage APIs allow web applications to store data locally.

**localStorage:**
- Persistent storage (survives browser restart)
- 5-10MB limit
- Domain-specific
- String values only
- Synchronous API

**sessionStorage:**
- Session-only storage (cleared on tab close)
- Same API as localStorage
- Tab-specific
- String values only

**IndexedDB:**
- Large object storage
- Asynchronous API
- Complex data structures
- More powerful than localStorage

**Cookies:**
- Small data (4KB)
- Sent with requests
- Can expire
- Domain/path scoped

**Use Cases:**
- User preferences
- Shopping cart
- Form data
- Caching
- Offline data`,
					CodeExamples: `// localStorage
// Set
localStorage.setItem("username", "john");
localStorage.setItem("theme", "dark");

// Get
const username = localStorage.getItem("username");
const theme = localStorage.getItem("theme") || "light";

// Remove
localStorage.removeItem("username");

// Clear all
localStorage.clear();

// Store objects
const user = { name: "John", age: 30 };
localStorage.setItem("user", JSON.stringify(user));
const storedUser = JSON.parse(localStorage.getItem("user"));

// Check if available
if (typeof Storage !== "undefined") {
    // localStorage is supported
}

// sessionStorage (same API)
sessionStorage.setItem("sessionId", "abc123");
const sessionId = sessionStorage.getItem("sessionId");

// Cookies
function setCookie(name, value, days) {
    const expires = new Date();
    expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
    document.cookie = name + '=' + value + ';expires=' + expires.toUTCString() + ';path=/';
}

function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(";");
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === " ") c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) === 0) {
            return c.substring(nameEQ.length, c.length);
        }
    }
    return null;
}`,
				},
				{
					Title: "Fetch API and HTTP Requests",
					Content: `The Fetch API provides a modern way to make HTTP requests.

**Fetch API:**
- Promise-based
- Modern replacement for XMLHttpRequest
- Built into browsers
- Supports async/await

**HTTP Methods:**
- GET: Retrieve data
- POST: Create data
- PUT: Update data
- DELETE: Delete data
- PATCH: Partial update

**Request Options:**
- method: HTTP method
- headers: Request headers
- body: Request body
- mode: CORS mode
- credentials: Include cookies

**Error Handling:**
- Network errors
- HTTP errors (4xx, 5xx)
- JSON parsing errors
- Timeout handling`,
					CodeExamples: `// Basic GET request
fetch("https://api.example.com/data")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error("Error:", error));

// With async/await
async function fetchData() {
    try {
        const response = await fetch("https://api.example.com/data");
        if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error:", error);
    }
}

// POST request
async function postData(url, data) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });
    return response.json();
}

// With authentication
fetch("https://api.example.com/protected", {
    headers: {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    },
});

// Error handling
async function safeFetch(url) {
    try {
        const response = await fetch(url);
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error("Not found");
            } else if (response.status === 500) {
                throw new Error("Server error");
            }
            throw new Error('HTTP ' + response.status);
        }
        
        const data = await response.json();
        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Timeout
function fetchWithTimeout(url, timeout = 5000) {
    return Promise.race([
        fetch(url),
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), timeout)
        ),
    ]);
}`,
				},
				{
					Title: "Geolocation and Notification APIs",
					Content: `Geolocation and Notification APIs enable location-based features and user notifications.

**Geolocation API:**
- getCurrentPosition(): Get current location
- watchPosition(): Watch position changes
- Requires user permission
- Returns coordinates (latitude, longitude)
- Can include accuracy, altitude, heading, speed

**Notification API:**
- Request permission first
- Create notifications
- Show persistent notifications
- Handle notification events
- Useful for PWAs

**Clipboard API:**
- writeText(): Copy to clipboard
- readText(): Read from clipboard
- Requires user interaction
- Modern replacement for execCommand

**Best Practices:**
- Always request permission
- Handle permission denied
- Provide fallbacks
- Respect user privacy`,
					CodeExamples: `// Geolocation API
if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition(
        (position) => {
            console.log("Latitude:", position.coords.latitude);
            console.log("Longitude:", position.coords.longitude);
            console.log("Accuracy:", position.coords.accuracy, "meters");
        },
        (error) => {
            console.error("Error:", error.message);
        },
        {
            enableHighAccuracy: true,
            timeout: 5000,
            maximumAge: 0
        }
    );
}

// Notification API
async function showNotification(title, options) {
    if ("Notification" in window) {
        const permission = await Notification.requestPermission();
        if (permission === "granted") {
            new Notification(title, options);
        }
    }
}

// Clipboard API
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        console.log("Copied!");
    } catch (err) {
        console.error("Failed:", err);
    }
}`,
				},
			},
			ProblemIDs: []int{},
		},
		{
			ID:          209,
			Title:       "ES6+ Features",
			Description: "Modern JavaScript features: let/const, arrow functions, destructuring, modules, and more.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "ES6 Core Features",
					Content: `ES6 (ES2015) introduced major improvements to JavaScript.

**let and const:**
- let: Block-scoped, can be reassigned
- const: Block-scoped, cannot be reassigned
- Replaces var (avoid var in modern code)
- Temporal dead zone

**Arrow Functions:**
- Shorter syntax
- Lexical this binding
- Implicit return
- Cannot be used as constructors

**Template Literals:**
- Backticks for strings
- Variable interpolation with ${}
- Multi-line strings
- Tagged templates

**Destructuring:**
- Extract values from arrays/objects
- Default values
- Rest/spread operators
- Nested destructuring

**Default Parameters:**
- Function parameter defaults
- Evaluated at call time
- Can use previous parameters`,
					CodeExamples: `// let and const
let count = 0;
count = 1; // OK

const PI = 3.14159;
// PI = 3.14; // Error: Assignment to constant

// Block scope
{
    let x = 1;
    const y = 2;
}
// console.log(x); // Error: x is not defined

// Arrow functions
const add = (a, b) => a + b;

const greet = name => 'Hello, ' + name;

const process = (data) => {
    // Multiple statements
    const result = data.map(x => x * 2);
    return result.filter(x => x > 10);
};

// Template literals
const name = "John";
const message = 'Hello, ' + name + '!\nThis is a multi-line\nstring.';

const calculation = '2 + 2 = ' + (2 + 2);

// Destructuring arrays
const [first, second, ...rest] = [1, 2, 3, 4, 5];
// first = 1, second = 2, rest = [3, 4, 5]

// Destructuring objects
const person = { name: "John", age: 30, city: "NYC" };
const { name, age } = person;
const { name: personName, age: personAge } = person;

// Default values
const greetUser = (name = "Guest") => 'Hello, ' + name;

function createUser({ name, age = 18, city = "Unknown" }) {
    return { name, age, city };
}`,
				},
				{
					Title: "ES6+ Advanced Features",
					Content: `Modern JavaScript continues to evolve with new features.

**Spread and Rest Operators:**
- ...spread: Expand arrays/objects
- ...rest: Collect remaining items
- Shallow copying
- Merging objects/arrays

**Modules:**
- import/export syntax
- Named exports
- Default exports
- Dynamic imports

**Promises and Async/Await:**
- Promise: Asynchronous operations
- async/await: Syntactic sugar
- Error handling with try/catch
- Promise.all(), Promise.race()

**Classes:**
- Class syntax
- Constructor
- Methods
- Inheritance with extends
- Static methods

**Map and Set:**
- Map: Key-value pairs
- Set: Unique values
- WeakMap, WeakSet`,
					CodeExamples: `// Spread operator
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2];

const obj1 = { a: 1, b: 2 };
const obj2 = { c: 3, d: 4 };
const merged = { ...obj1, ...obj2 };

// Rest operator
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}

// Modules
// math.js
export const add = (a, b) => a + b;
export const subtract = (a, b) => a - b;
export default function multiply(a, b) {
    return a * b;
}

// main.js
import multiply, { add, subtract } from "./math.js";

// Dynamic import
const module = await import("./math.js");

// Promises
const promise = new Promise((resolve, reject) => {
    setTimeout(() => resolve("Success"), 1000);
});

promise
    .then(result => console.log(result))
    .catch(error => console.error(error));

// Async/await
async function fetchData() {
    try {
        const response = await fetch("/api/data");
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(error);
    }
}

// Promise.all
const promises = [fetch("/api/1"), fetch("/api/2"), fetch("/api/3")];
const results = await Promise.all(promises);

// Classes
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    greet() {
        return 'Hello, I\'m ' + this.name;
    }
    
    static create(name) {
        return new Person(name, 0);
    }
}

class Student extends Person {
    constructor(name, age, school) {
        super(name, age);
        this.school = school;
    }
}

// Map and Set
const map = new Map();
map.set("key", "value");
map.get("key");

const set = new Set([1, 2, 3, 3]); // {1, 2, 3}`,
				},
				{
					Title: "ES6 Modules and Iterators",
					Content: `ES6 modules provide a way to organize and share code across files.

**ES6 Modules:**
- import: Import from other modules
- export: Export from module
- Named exports: Multiple exports
- Default export: Single default export
- Static analysis: Better tooling support

**Module Syntax:**
- import { name } from "./module.js"
- import defaultName from "./module.js"
- export const name = value
- export default value

**Iterators and Generators:**
- Iterators: Objects with next() method
- Generators: Functions that yield values
- for...of: Iterate over iterables
- Symbol.iterator: Make objects iterable

**Use Cases:**
- Code organization
- Dependency management
- Tree shaking
- Lazy loading`,
					CodeExamples: `// Named exports (math.js)
export const add = (a, b) => a + b;
export const subtract = (a, b) => a - b;
export const PI = 3.14159;

// Default export (math.js)
export default function multiply(a, b) {
    return a * b;
}

// Importing (main.js)
import multiply, { add, subtract, PI } from "./math.js";

// Or import all
import * as math from "./math.js";
math.add(1, 2);

// Dynamic import
const module = await import("./math.js");

// Iterators
const numbers = [1, 2, 3];
const iterator = numbers[Symbol.iterator]();

console.log(iterator.next()); // { value: 1, done: false }
console.log(iterator.next()); // { value: 2, done: false }
console.log(iterator.next()); // { value: 3, done: false }
console.log(iterator.next()); // { value: undefined, done: true }

// Custom iterator
const range = {
    start: 1,
    end: 5,
    [Symbol.iterator]() {
        let current = this.start;
        let end = this.end;
        return {
            next() {
                if (current <= end) {
                    return { value: current++, done: false };
                }
                return { done: true };
            }
        };
    }
};

for (const num of range) {
    console.log(num); // 1, 2, 3, 4, 5
}

// Generators
function* numberGenerator() {
    yield 1;
    yield 2;
    yield 3;
}

const gen = numberGenerator();
console.log(gen.next()); // { value: 1, done: false }
console.log(gen.next()); // { value: 2, done: false }

// Generator with parameters
function* fibonacci() {
    let [prev, curr] = [0, 1];
    while (true) {
        yield curr;
        [prev, curr] = [curr, prev + curr];
    }
}

const fib = fibonacci();
console.log(fib.next().value); // 1
console.log(fib.next().value); // 1
console.log(fib.next().value); // 2`,
				},
			},
			ProblemIDs: []int{},
		},
	})
}
