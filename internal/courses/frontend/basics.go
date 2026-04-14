package frontend

import "github.com/rusik69/lc/internal/problems"

func init() {
	problems.RegisterFrontendModules([]problems.CourseModule{
		{
			ID:          1500,
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
					Content: `HTML elements are the building blocks of every web page you encounter on the internet. Think of them as the individual LEGO bricks that snap together to form a complete structure. Each element typically consists of three parts: an opening tag that marks where the element begins, the content that lives inside, and a closing tag that signals the end. Some special elements, known as void or self-closing elements (like img and br), do not require a closing tag because they do not contain nested content. Understanding elements deeply is the first step to becoming a proficient web developer, because every single thing you see on a web page -- every heading, paragraph, image, link, and button -- is represented by an HTML element.

**1. Headings -- The Organizational Backbone of Your Content:**

HTML provides six levels of headings, from h1 (the most important and typically the largest) down to h6 (the least prominent). Headings are not merely about visual size -- they create a semantic hierarchy that tells browsers, search engines, and screen readers how your content is organized. Think of headings like the outline of a research paper: h1 is the title, h2 elements are major sections, h3 elements are subsections within those, and so on. A well-structured heading hierarchy is critical for accessibility because screen reader users often navigate pages by jumping between headings. It also matters for SEO, as search engines use heading structure to understand what your page is about. Best practice dictates using exactly one h1 per page (the main topic) and never skipping levels (for example, jumping from h2 directly to h4).

**2. Paragraphs -- The Workhorse of Text Content:**

The p element is how you wrap blocks of text on a web page. Browsers automatically add vertical spacing (margin) above and below paragraphs, visually separating them from surrounding content. This is important because, unlike a word processor, simply pressing Enter in your HTML source code does not create a new line in the rendered output -- HTML collapses whitespace. You must use p tags to logically group sentences into paragraphs. Beyond visual formatting, the p element carries semantic meaning: it tells assistive technologies that a block of text forms a coherent thought or idea, allowing screen readers to pause appropriately between paragraphs for a more natural reading experience.

**3. Links -- The Connective Tissue of the Web:**

The anchor element (a) with its href attribute is what makes the "HyperText" in HTML possible. Links are the fundamental mechanism that connects billions of web pages into the interconnected web we use every day. Links can point to external websites, internal pages within your own site, specific sections of a page (using anchor/hash links), email addresses (mailto: links), or even telephone numbers (tel: links). The text inside a link is called the "anchor text," and it should be descriptive enough that users (and search engines) understand where the link leads without needing additional context. Avoid generic anchor text like "click here" -- instead, use meaningful descriptions like "read our accessibility guidelines." Links are also a cornerstone of SEO, as search engines evaluate both incoming and outgoing links to determine page authority and relevance.

**4. Images -- Visual Content with Accessibility Built In:**

The img element embeds images into your page and is one of HTML's void elements (it has no closing tag). Two attributes are essential: src (the path or URL to the image file) and alt (alternative text describing the image). The alt attribute is not optional -- it is a critical accessibility feature that provides a text description for users who cannot see the image, whether due to visual impairment, slow network connections, or broken image links. Screen readers read alt text aloud, so it should be concise yet descriptive. For purely decorative images that add no informational value, use an empty alt attribute (alt="") to signal screen readers to skip them. Modern HTML also supports responsive images through the srcset and sizes attributes, allowing browsers to choose the most appropriate image size based on the user's device and viewport.

**5. Lists -- Structuring Related Items:**

HTML offers three types of lists. Unordered lists (ul) display items with bullet points and are used when the order of items does not matter -- like a shopping list or a set of features. Ordered lists (ol) display numbered items and are appropriate when sequence matters -- like step-by-step instructions or rankings. Each item within either list type is wrapped in an li element. There are also description lists (dl) with dt (term) and dd (description) pairs, useful for glossaries or key-value displays. Lists are not just visual -- they carry semantic meaning. Screen readers announce "list of 5 items" when encountering a ul, helping users understand the structure of content. Nesting lists inside other lists creates sub-hierarchies, which is useful for multi-level navigation menus or complex outlines.

**6. Semantic Elements (HTML5) -- Meaning Beyond Structure:**

HTML5 introduced a set of semantic elements that go beyond generic div containers to convey the purpose and role of content sections. The header element represents introductory content or navigational aids. The nav element wraps navigation links. The main element identifies the dominant content of the page (only one per page). The article element represents a self-contained composition that could be independently distributed -- like a blog post or news story. The section element groups thematically related content. The aside element contains content tangentially related to the surrounding content, like sidebars. The footer element represents footer content for its nearest sectioning ancestor. Using these elements instead of generic divs provides enormous benefits: screen readers can offer landmark-based navigation, search engines better understand your content structure, your code becomes self-documenting, and CSS styling becomes more intuitive because you can target meaningful element names instead of arbitrary class names.`,
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
					Content: `Attributes are the mechanism through which HTML elements receive additional configuration, metadata, and behavioral instructions. If HTML elements are the nouns of a web page, attributes are the adjectives and adverbs that describe and modify them. Every attribute is specified within the opening tag of an element, following the pattern name="value". Understanding attributes is essential because they control everything from where a link navigates to, what image gets displayed, how form data is submitted, and how assistive technologies interpret your content. Attributes are the bridge between the static structure of HTML and the dynamic, interactive web experiences users expect.

**1. Global Attributes -- Universal Modifiers for Any Element:**

Global attributes can be applied to any HTML element, making them incredibly versatile. The id attribute assigns a unique identifier to an element -- no two elements on the same page should share the same id. This uniqueness makes id perfect for JavaScript targeting (getElementById), CSS styling of specific elements, and creating anchor links that jump to particular page sections. The class attribute, in contrast, can be shared across multiple elements and is the primary hook for CSS styling. You can apply multiple classes to a single element by separating them with spaces, enabling a composable approach to styling. The style attribute allows inline CSS directly on an element, but this practice is generally discouraged because it mixes presentation with structure and makes maintenance harder. The title attribute provides tooltip text that appears when users hover over an element. Perhaps most powerful are data-* attributes (data-user-id, data-price, data-status, etc.), which let you embed custom data directly in your HTML that JavaScript can later read and manipulate. The lang attribute specifies the language of an element's content, which is critical for screen readers to use correct pronunciation and for search engines to understand content language.

**2. Element-Specific Attributes -- Tailored Configuration:**

Many attributes only make sense on specific elements. The href attribute on anchor (a) elements specifies where the link navigates -- it can be an absolute URL, a relative path, a hash link to a page section, or even a mailto: or tel: link. The src attribute on img, script, and iframe elements points to the resource to load. The alt attribute on img elements provides alternative text for accessibility -- this is one of the most important attributes in all of HTML, as it ensures visually impaired users can understand image content through screen readers. For form elements, the type attribute on input determines the kind of data expected (text, email, password, number, date, checkbox, radio, etc.), each rendering a different control in the browser. The name attribute identifies form controls when data is submitted to a server. The value attribute sets a default value. Boolean attributes like required, disabled, and readonly modify form behavior: required prevents submission until the field is filled, disabled grays out the control and excludes it from form data, and readonly allows viewing but not editing. The placeholder attribute shows hint text inside empty inputs, while autocomplete helps browsers auto-fill form fields based on previously entered data.

**3. ARIA Attributes -- Bridging Accessibility Gaps:**

When native HTML semantics are not sufficient to convey an element's purpose to assistive technologies, ARIA (Accessible Rich Internet Applications) attributes fill the gap. Attributes like aria-label provide an accessible name when visible text is absent, aria-describedby references a separate element that provides additional description, aria-hidden="true" hides decorative elements from screen readers, and role overrides or clarifies an element's semantic role. While native semantic HTML should always be your first choice, ARIA attributes are indispensable for complex interactive widgets like custom dropdowns, modals, tabs, and accordions.

**4. Best Practices for Working with Attributes:**

Always include alt text on images -- this is both an accessibility requirement and an SEO benefit. Use meaningful, semantic class names that describe what an element is (like "navigation-menu" or "product-card") rather than how it looks (like "blue-box" or "left-column"). Avoid inline styles via the style attribute; instead, use external or internal stylesheets for maintainability. Leverage data-* attributes for storing custom data rather than misusing existing attributes or inventing non-standard ones. Always quote attribute values, even when technically optional, for consistency and to avoid parsing errors. When building forms, always associate label elements with their corresponding inputs using the for attribute -- this is critical for both accessibility and usability, as clicking the label focuses the associated input.`,
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
					Content: `Forms and tables represent two of the most practical and frequently used categories of HTML elements. Forms are the primary mechanism through which users interact with web applications -- every login screen, search bar, checkout page, registration form, and contact form relies on HTML form elements. Tables, on the other hand, are designed specifically for displaying structured, tabular data -- think spreadsheets, financial reports, comparison charts, and schedule displays. Mastering both is essential for building functional, data-driven web applications.

**1. HTML Forms -- The Gateway to User Interaction:**

The form element serves as a container for all the interactive controls that collect user input. Think of it as an envelope: it wraps up all the data fields and, when submitted, sends that data to a server for processing. The action attribute specifies the URL where the form data should be sent, while the method attribute determines how it gets there. GET appends data to the URL as query parameters (visible in the address bar, suitable for searches and bookmarkable actions), while POST sends data in the request body (hidden from the URL, appropriate for sensitive data like passwords or large payloads). The enctype attribute becomes important when uploading files -- you must set it to "multipart/form-data" to properly transmit file contents.

**2. Form Controls -- The Individual Input Mechanisms:**

The input element is the most versatile form control, with its behavior determined by the type attribute. A text input creates a single-line text field, email validates email format, password masks typed characters, number restricts to numeric values with optional min/max constraints, and date renders a native date picker. Beyond input, the textarea element provides multi-line text entry (perfect for comments or messages), the select element creates dropdown menus with option children, and the button element triggers form submission or custom actions. The label element is critically important for accessibility -- it associates descriptive text with a form control, allowing screen reader users to understand what each field expects. Labels can be linked via the "for" attribute matching an input's "id", or by wrapping the input element directly. The fieldset and legend elements group related form controls visually and semantically, which is particularly useful for radio button groups and checkbox sets.

**3. HTML Tables -- Structured Data Display:**

Tables in HTML are built from a hierarchy of elements that mirror the rows-and-columns structure of a spreadsheet. The table element is the outer container. Inside it, thead wraps the header row(s), tbody contains the main data rows, and tfoot holds footer rows (often used for totals or summaries). Each row is defined by a tr element. Within rows, th elements define header cells (typically bold and centered by default) while td elements contain regular data cells. The colspan attribute lets a cell span multiple columns, and rowspan lets it span multiple rows -- useful for merged cells in complex reports. The scope attribute on th elements (with values "col", "row", "colgroup", or "rowgroup") tells assistive technologies whether the header applies to a column, row, or group thereof, which is essential for screen reader users navigating table data.

**4. Best Practices and Common Pitfalls:**

Always include a caption element inside your table to provide a title or summary -- this benefits both sighted users who need context and screen reader users who need to understand the table's purpose before diving into the data. Use the scope attribute on every th to ensure proper accessibility. Never use tables for page layout -- this was a common practice in the 1990s and early 2000s, but modern CSS (Grid and Flexbox) provides far superior layout tools that are more accessible, maintainable, and flexible. For forms, always validate data on both the client side (for user experience) and the server side (for security, since client-side validation can be bypassed). Group related fields with fieldset elements and always provide clear, visible labels for every input.`,
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
					Content: `Multimedia elements and meta tags serve two distinct but equally important purposes in modern web development. Multimedia elements -- images, video, and audio -- transform web pages from static text documents into rich, engaging experiences that can rival native applications. Meta tags, while invisible to users viewing the page, provide crucial instructions to browsers, search engines, and social media platforms about how to interpret, display, and share your content. Together, they represent the difference between a bare-bones text page and a fully realized, professional web presence.

**1. Multimedia Elements -- Bringing Pages to Life:**

The img element is the most common multimedia element, embedding images directly into your page. As a void element (no closing tag), it requires at minimum the src attribute (pointing to the image file) and the alt attribute (providing descriptive text for accessibility). HTML5 dramatically expanded multimedia capabilities with the video and audio elements, which provide native playback controls without requiring third-party plugins like Flash (which is now defunct). Both elements support the controls attribute to display play/pause/volume buttons, and you can nest multiple source elements inside them to offer different file formats -- browsers will use the first compatible format they find. The picture element takes responsive images further by allowing you to define entirely different image sources for different viewport sizes or screen resolutions, giving you fine-grained control over which image loads on which device. The iframe element embeds external content like YouTube videos, Google Maps, or other web pages, though it should be used judiciously due to performance and security considerations.

**2. Meta Tags -- The Invisible Instructions:**

Meta tags live inside the head section of your HTML document and provide metadata that browsers, search engines, and social platforms use to properly handle your page. The charset meta tag (almost always set to UTF-8) specifies character encoding, ensuring that special characters, accented letters, and emoji render correctly across all browsers and operating systems. The viewport meta tag is absolutely essential for mobile-responsive design -- without it, mobile browsers will render your page at a desktop width and then shrink it to fit, resulting in tiny, unreadable text. Setting it to "width=device-width, initial-scale=1.0" tells the browser to match the page width to the device's screen width. The description meta tag provides a summary of your page that search engines typically display in search results -- a compelling description can significantly improve click-through rates. While the keywords meta tag has been largely deprecated by search engines (because it was heavily abused for keyword stuffing), it is still occasionally used in niche contexts.

**3. Social Media and Open Graph Tags:**

When someone shares a link to your page on Facebook, LinkedIn, Twitter, or Slack, those platforms look for Open Graph (og:) and Twitter Card meta tags to determine what title, description, and image to display in the preview card. Without these tags, platforms will attempt to guess, often with poor results -- a missing image, a truncated or irrelevant description, or a generic title. Including og:title, og:description, og:image, and og:url tags ensures your content looks professional and engaging when shared. Twitter has its own set of card tags (twitter:card, twitter:title, twitter:description, twitter:image) that provide similar functionality specifically for that platform. Investing a few minutes to configure these tags can dramatically improve the shareability and professional appearance of your web pages across social media.

**4. Best Practices for Multimedia and Meta Tags:**

Always include descriptive alt text for every informational image -- this is both an accessibility requirement and an SEO advantage. Use responsive images with srcset and sizes attributes to serve appropriately sized images based on the user's device, reducing bandwidth usage and improving load times. Optimize media file sizes aggressively using modern formats like WebP for images and MP4/WebM for video. Include all essential meta tags (charset, viewport, description, Open Graph) on every page. When embedding video or audio, always provide captions or transcripts for accessibility, and consider using the loading="lazy" attribute on images that appear below the fold to improve initial page load performance.`,
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
			ID:          1501,
			Title:       "CSS Fundamentals",
			Description: "Master CSS styling: selectors, properties, values, and the cascade.",
			Order:       1,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to CSS",
					Content: `CSS (Cascading Style Sheets) is the language that transforms plain, unstyled HTML documents into visually appealing, professionally designed web pages. If HTML is the skeleton of a web page, CSS is the skin, clothing, and makeup -- it controls every visual aspect of how content appears to users, from colors and fonts to spacing, positioning, animations, and responsive layouts. CSS was invented in 1996 by Hakon Wium Lie and Bert Bos to solve a fundamental problem: the mixing of content and presentation in early HTML. Before CSS, developers used HTML tags like font and center to control appearance, which made code bloated, repetitive, and nearly impossible to maintain at scale.

**1. The Separation of Concerns -- Why CSS Exists:**

The core philosophy behind CSS is the separation of concerns: HTML should describe what content is (structure and meaning), while CSS should describe how content looks (presentation and layout). This separation provides enormous practical benefits. You can completely redesign a website's appearance without touching a single line of HTML. Multiple pages can share the same stylesheet, ensuring visual consistency and making site-wide changes trivial. Different stylesheets can be applied for different contexts -- one for screens, another for print, and yet another for accessibility preferences. This separation also enables team collaboration, where content writers and designers can work independently without stepping on each other's code.

**2. CSS Syntax -- The Rule-Based Approach:**

CSS works through rules. Each rule consists of a selector (which targets specific HTML elements), followed by a declaration block containing one or more property-value pairs. The selector acts like an address system, telling the browser which elements to style. The property specifies what aspect to change (color, font-size, margin, background, etc.), and the value specifies how to change it (red, 16px, 20px, blue, etc.). This rule-based approach is powerful because a single rule can affect hundreds of elements simultaneously. For example, a rule targeting all paragraph elements can set the font, color, and spacing for every paragraph on every page of your site with just a few lines of code.

**3. Three Ways to Add CSS -- From Quick Fixes to Production Standards:**

There are three methods for applying CSS to HTML. Inline styles use the style attribute directly on individual elements -- this is the most specific but least maintainable approach, suitable only for quick overrides or dynamic styles set by JavaScript. Internal stylesheets use a style tag in the HTML document's head section, which is convenient for single-page prototypes but does not scale across multiple pages. External stylesheets (separate .css files linked via a link tag) are the industry standard for production websites because they enable caching (the browser downloads the CSS once and reuses it across pages), maintain clean separation of concerns, and allow the same styles to be shared across an entire site. In professional development, you will almost exclusively use external stylesheets.

**4. CSS Selectors -- The Targeting System:**

Selectors are the mechanism by which CSS identifies which elements to style, and mastering them is key to writing efficient, maintainable CSS. Element selectors (like p, div, h1) target all instances of an HTML element. Class selectors (prefixed with a dot, like .highlight) target elements with a specific class attribute -- this is the most commonly used selector in professional CSS because classes are reusable and composable. ID selectors (prefixed with a hash, like #header) target a single unique element, but are generally avoided for styling because their high specificity makes overrides difficult. Attribute selectors (like [type="text"]) target elements based on their attributes. Pseudo-classes (like :hover, :focus, :first-child) target elements in specific states or positions. Pseudo-elements (like ::before, ::after, ::first-line) target specific parts of an element's content. Combinators allow you to target elements based on their relationship to other elements: descendant (space), child (>), adjacent sibling (+), and general sibling (~). Learning to combine these selectors effectively is one of the most important CSS skills you will develop.`,
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
					Content: `CSS properties are the individual instructions that control every visual aspect of an element's appearance on screen. There are hundreds of CSS properties, but a core set handles the vast majority of everyday styling needs. Understanding these properties deeply -- not just their names but how they interact, which values they accept, and how they behave in different contexts -- is what separates a novice from a proficient CSS developer. Think of properties as the dials and knobs on a mixing board: each controls one specific dimension of the output, and the art lies in combining them effectively.

**1. Text Properties -- Controlling Typography:**

Typography is arguably the most important aspect of web design, since the majority of web content is text. The color property sets text color and accepts any valid CSS color value. The font-family property specifies which typeface to use, and you should always provide a fallback stack (like 'Helvetica Neue', Arial, sans-serif) so browsers have alternatives if the primary font is unavailable. The font-size property controls text size and can use various units (px, em, rem, %, vw). The font-weight property controls boldness on a scale from 100 (thinnest) to 900 (boldest), with "normal" being 400 and "bold" being 700. The text-align property controls horizontal alignment (left, center, right, justify). The text-decoration property manages underlines, overlines, and strikethroughs -- most commonly used to remove underlines from links or add them for emphasis. The line-height property controls the vertical spacing between lines of text and is crucial for readability; a value of 1.5 to 1.6 is generally considered optimal for body text. The letter-spacing and word-spacing properties provide fine-tuned control over character and word spacing.

**2. The Box Model -- The Foundation of CSS Layout:**

Every single HTML element on a web page is rendered as a rectangular box, and the box model describes the layers of that box. At the center is the content area (controlled by width and height), where text and child elements live. Surrounding the content is padding -- the space between the content and the element's border. This is like the internal cushioning of a picture frame. Next comes the border itself, which can be styled with width, color, and pattern (solid, dashed, dotted, etc.). Finally, the margin creates space outside the border, pushing other elements away -- like the gap between picture frames on a wall. A critical concept is the box-sizing property: by default (content-box), width and height apply only to the content area, and padding/border are added on top, making the total element size larger than specified. Setting box-sizing to border-box includes padding and border in the width/height calculation, which is far more intuitive and is why most modern CSS resets apply border-box to all elements.

**3. Color Values -- Multiple Ways to Express Color:**

CSS offers several color systems, each with different strengths. Named colors (red, blue, cornflowerblue) are readable but limited to 147 predefined options. Hexadecimal codes (#FF0000 for red, #00FF00 for green) are compact and widely used -- each pair of hex digits represents the red, green, and blue channels (0-255). RGB notation (rgb(255, 0, 0)) is functionally equivalent to hex but more human-readable. RGBA adds an alpha channel for transparency (rgba(255, 0, 0, 0.5) is 50% transparent red), which is invaluable for overlays, shadows, and subtle design effects. HSL (Hue, Saturation, Lightness) is often preferred by designers because it maps more intuitively to how humans perceive color: hue is the color wheel position (0-360 degrees), saturation is the intensity (0-100%), and lightness is how bright or dark (0-100%). Modern CSS also supports the newer color functions like oklch() and color() for wider gamut displays.

**4. CSS Units -- Absolute vs. Relative:**

Understanding CSS units is essential for building responsive, accessible designs. Pixels (px) are absolute units that provide precise control but do not adapt to user preferences or screen sizes. Em units are relative to the font-size of the parent element, making them useful for component-scoped sizing that scales proportionally. Rem units (root em) are relative to the root element's font-size (typically the html element, defaulting to 16px), providing consistent scaling across the entire page and respecting user browser font-size preferences -- this is why rem is the preferred unit for modern responsive typography. Percentages are relative to the parent element's corresponding property (width: 50% means half the parent's width). Viewport units (vw and vh) represent percentages of the browser window's width and height, perfect for full-screen hero sections or viewport-responsive typography. The modern clamp() function combines minimum, preferred, and maximum values (like clamp(1rem, 2.5vw, 2rem)) for fluid, responsive sizing without media queries.`,
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
					Content: `Understanding CSS layout is the gateway to creating professional, well-organized web pages that adapt gracefully to different screen sizes. Layout is arguably the most challenging aspect of CSS for beginners, because it involves understanding how the browser positions elements in relation to each other and to the viewport. Before modern layout tools like Flexbox and Grid existed, developers relied on floats, tables, and absolute positioning -- techniques that were often described as "fighting CSS." Today, CSS provides powerful, intuitive layout mechanisms, but understanding the fundamental concepts behind element flow and positioning remains essential.

**1. The Display Property -- Controlling Element Behavior:**

The display property is the single most important property for understanding CSS layout because it determines how an element participates in the page flow. Block-level elements (display: block) behave like rectangular boxes that stretch to fill the full width of their container and always start on a new line -- think of them as paragraphs, each stacking vertically on top of the next. Divs, paragraphs, headings, and list items are block-level by default. Inline elements (display: inline) flow within text, taking up only as much width as their content requires and sitting side-by-side on the same line -- spans, links, and emphasis elements are inline by default. Inline-block (display: inline-block) combines both behaviors: elements sit on the same line like inline elements but can have width, height, padding, and margins applied like block elements. Setting display to none completely removes an element from the page flow, as if it does not exist in the document at all. Most importantly for modern development, display: flex creates a Flexbox container and display: grid creates a Grid container -- these two values unlock the most powerful layout systems in CSS.

**2. The Position Property -- Breaking Out of Normal Flow:**

By default, all elements follow "normal flow" -- block elements stack vertically and inline elements flow horizontally within blocks. The position property lets you override this behavior. Static positioning is the default and means the element follows normal flow. Relative positioning offsets the element from its normal position using top, right, bottom, and left properties, but crucially, the original space the element occupied is preserved -- other elements behave as if the element had not moved. Absolute positioning removes the element entirely from normal flow and positions it relative to its nearest ancestor that has a non-static position (its "containing block"). If no such ancestor exists, it positions relative to the viewport. This is incredibly useful for overlays, tooltips, dropdown menus, and decorative elements. Fixed positioning is similar to absolute, but the element is positioned relative to the browser viewport and stays in place even when the user scrolls -- perfect for sticky headers, floating action buttons, and persistent navigation bars. Sticky positioning is a hybrid: the element behaves as relatively positioned until it reaches a specified scroll threshold, at which point it "sticks" in place like a fixed element -- ideal for table headers that remain visible while scrolling through long data sets.

**3. Float -- The Legacy Layout Tool:**

Before Flexbox and Grid, the float property was the primary tool for creating multi-column layouts. Float was originally designed for a single purpose: allowing text to wrap around images, like in a newspaper article. However, creative developers repurposed it for entire page layouts, leading to complex and fragile code. A floated element is taken out of normal flow and pushed to the left or right of its container, with surrounding content flowing around it. While float still has legitimate uses for wrapping text around images, it should no longer be used for general page layout. One common gotcha with floats is that parent containers collapse when all their children are floated -- the "clearfix" hack was invented to solve this problem, but modern developers simply use Flexbox or Grid instead.

**4. Common Layout Patterns:**

Centering content is one of the most frequently needed layout tasks. Horizontally centering a block element requires setting a width and applying margin: 0 auto. Vertically centering historically required hacks, but Flexbox makes it trivial with display: flex; justify-content: center; align-items: center. Two-column and three-column layouts are easily achieved with Flexbox (flex containers with flex items) or Grid (grid-template-columns). The classic header-content-footer layout can be built with Flexbox on the body using flex-direction: column and flex: 1 on the main content area, ensuring the footer always sticks to the bottom of the viewport even when content is short. Navigation bars are typically built as Flex containers with justify-content: space-between to distribute items evenly.`,
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
					Content: `The "Cascading" in Cascading Style Sheets refers to the algorithm browsers use to determine which CSS rules win when multiple rules target the same element with conflicting property values. Understanding specificity and the cascade is essential for writing predictable, maintainable CSS -- without this knowledge, you will inevitably find yourself frustrated by styles that mysteriously refuse to apply, or worse, reaching for the !important keyword as a sledgehammer solution to every conflict. Think of the cascade as a dispute resolution system: when multiple rules claim authority over the same property of the same element, the cascade determines which rule has the highest priority.

**1. Specificity -- The Weighted Scoring System:**

Specificity is a numerical scoring system that determines which CSS rule takes precedence when multiple rules target the same element. Every CSS selector has a specificity score calculated from four categories, often represented as a comma-separated tuple (a, b, c, d). Inline styles (applied via the style attribute) have the highest specificity at (1, 0, 0, 0), which is why they override almost everything. ID selectors contribute to the second category at (0, 1, 0, 0) per ID in the selector. Class selectors, attribute selectors, and pseudo-classes each contribute (0, 0, 1, 0). Element selectors and pseudo-elements contribute (0, 0, 0, 1). The universal selector (*) contributes nothing (0, 0, 0, 0). When comparing specificity scores, the browser compares each category from left to right: a single ID selector (0, 1, 0, 0) will always beat any number of class selectors (0, 0, 99, 0), because the second column outranks the third. This is why professional CSS developers prefer class-based selectors -- they provide a predictable, manageable level of specificity that is easy to override when needed.

**2. The Cascade Algorithm -- How Browsers Resolve Conflicts:**

When multiple CSS rules apply conflicting values to the same property of the same element, the browser follows a precise cascade algorithm. First, it checks importance: rules marked with !important override all non-important rules regardless of specificity. Second, it checks origin: user-agent (browser default) styles have the lowest priority, followed by author (developer) styles, then user styles. Third, it compares specificity scores. Fourth, if specificity is equal, the rule that appears last in source order wins. This means that the order in which you write your CSS rules -- or the order in which stylesheets are linked in your HTML -- matters when specificity is tied. Understanding this sequence helps you predict exactly which rule will apply in any situation and eliminates the guesswork that plagues many CSS developers.

**3. The !important Declaration -- The Nuclear Option:**

The !important declaration forces a property value to override all other declarations regardless of specificity. While it can solve immediate problems, it creates a "specificity arms race" that makes CSS increasingly difficult to maintain over time. Once you start using !important, the only way to override it is with another !important on a more specific selector, leading to an escalating cycle of overrides. There are very few legitimate uses for !important: utility classes in design systems (like .hidden { display: none !important; }), overriding third-party CSS you cannot modify, and certain accessibility styles. In all other cases, the correct solution is to restructure your selectors so the desired rule naturally has higher specificity or later source order.

**4. Best Practices for Manageable Specificity:**

Keep your specificity as low and flat as possible. Prefer class selectors over IDs for styling, because a single ID in a selector immediately outranks any combination of classes. Use naming conventions like BEM (Block, Element, Modifier) to create descriptive, unique class names that avoid conflicts without high specificity. Organize your CSS from general to specific: base/reset styles first, then component styles, then utility overrides. Consider methodologies like ITCSS (Inverted Triangle CSS) that explicitly structure your stylesheet layers by specificity. When you find yourself in a specificity battle, resist the urge to add !important -- instead, review your selectors and simplify the one that needs to win. Modern tools like CSS-in-JS libraries and CSS Modules can help avoid specificity issues altogether by scoping styles to specific components.`,
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
					Content: `Every web browser ships with its own built-in default stylesheet (called the "user-agent stylesheet") that applies basic styling to HTML elements -- margins on paragraphs, bold text on headings, bullet points on list items, and so on. The problem is that these defaults vary slightly between browsers. Chrome, Firefox, Safari, and Edge each have their own ideas about how much margin a paragraph should have, what font size an h1 should be, or how a form input should look. These inconsistencies can cause your carefully crafted design to look slightly different across browsers. CSS resets and normalizations exist to solve this problem by establishing a consistent, predictable baseline before you start writing your own styles. Think of them as "leveling the playing field" -- ensuring every browser starts from the same point.

**1. CSS Reset -- The Clean Slate Approach:**

A CSS reset strips away all browser default styles, giving you a completely blank canvas to work with. Eric Meyer's Reset CSS (published in 2007 and still widely referenced) is the most famous example -- it removes margins, padding, borders, and font sizes from all elements, effectively making everything look like plain, unstyled text. The advantage is total control: you define every single style explicitly, with no surprises from browser defaults. The disadvantage is that you must restyle everything, including basic typographic elements that had perfectly reasonable defaults. Without restoring styles for headings, lists, and other elements, your page initially looks like an undifferentiated wall of text. For this reason, raw resets can be overkill for many projects and require more CSS work upfront.

**2. Normalize.css -- The Preservation Approach:**

Normalize.css, created by Nicolas Gallagher and Jonathan Neal, takes a fundamentally different philosophy: rather than stripping all defaults, it corrects inconsistencies between browsers while preserving useful defaults. It fixes known bugs (like display inconsistencies for HTML5 elements in older browsers), normalizes styles for elements that vary across browsers (like form inputs and tables), and adds minor improvements (like setting the default box-sizing). Because it preserves sensible defaults, you write less CSS to achieve a polished result. Normalize.css is generally the better choice for most projects because it does not fight against browser conventions -- it harmonizes them. It has been adopted by major projects including Bootstrap, HTML5 Boilerplate, and Twitter.

**3. Modern CSS Resets -- The Best of Both Worlds:**

Modern CSS resets, such as Andy Bell's "A Modern CSS Reset" and Josh Comeau's "Custom CSS Reset," combine the best ideas from both approaches. They typically include a small set of highly impactful rules: applying box-sizing: border-box to all elements (so width and height calculations include padding and border), removing default margins, setting a sensible body line-height, making images block-level and max-width 100% by default, inheriting fonts on form elements, and ensuring smooth scrolling behavior. These modern resets are minimal (often under 30 lines) yet address the most common pain points of cross-browser development. They pair perfectly with CSS custom properties (variables) for defining a consistent design token system -- colors, spacing, fonts, and breakpoints defined once and reused throughout your stylesheet.

**4. Choosing the Right Approach for Your Project:**

For most projects, a modern minimal reset combined with your own base styles is the best approach. Start by setting box-sizing: border-box globally (the single most impactful reset rule), remove default margins and padding, set a comfortable body font and line-height, make media elements responsive by default, and inherit fonts on form controls. If you are using a CSS framework like Tailwind CSS, Bootstrap, or Material UI, the framework typically includes its own reset or normalization, so you should not add a separate one. If you are building a design system or component library, consider a slightly more aggressive reset to minimize assumptions about the styling environment your components will live in. The key principle is consistency: whatever approach you choose, apply it to every project and make it the first thing in your stylesheet.`,
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
			ID:          1502,
			Title:       "JavaScript Basics",
			Description: "Learn JavaScript fundamentals: variables, functions, control flow, and data types.",
			Order:       2,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to JavaScript",
					Content: `JavaScript is the programming language of the web -- the only language that runs natively in every web browser, making it the essential tool for creating interactive, dynamic user experiences. While HTML provides structure and CSS provides styling, JavaScript provides behavior: it responds to user actions, manipulates page content in real time, communicates with servers, validates forms, creates animations, and powers everything from simple dropdown menus to complex single-page applications like Gmail, Google Maps, and Figma. Created in just 10 days in 1995 by Brendan Eich at Netscape, JavaScript has evolved from a simple scripting language for adding small interactive features into one of the most widely used and versatile programming languages in the world, running not just in browsers but also on servers (Node.js), mobile devices (React Native), desktops (Electron), and even embedded systems.

**1. What Makes JavaScript Unique Among Programming Languages:**

JavaScript is a high-level, dynamically typed, interpreted language that supports multiple programming paradigms -- you can write procedural code, object-oriented code, and functional code all in the same program. Unlike compiled languages like C or Java, JavaScript code is executed line by line at runtime by the browser's JavaScript engine (V8 in Chrome, SpiderMonkey in Firefox, JavaScriptCore in Safari). Dynamic typing means variables do not have fixed types -- a variable can hold a string one moment and a number the next. While this flexibility makes JavaScript easy to pick up quickly, it also introduces subtle bugs that statically typed languages catch at compile time, which is why TypeScript (a typed superset of JavaScript) has become extremely popular in professional development. JavaScript is also single-threaded, meaning it processes one operation at a time, but it uses an event loop and asynchronous patterns (callbacks, Promises, async/await) to handle concurrent operations like network requests without freezing the user interface.

**2. How JavaScript Integrates with HTML:**

There are three ways to include JavaScript in a web page. Inline scripts place JavaScript code directly within a script tag in the HTML document -- this is convenient for small scripts but mixes behavior with structure. External scripts reference a separate .js file via the src attribute on a script tag -- this is the professional standard because it promotes code reuse, caching, and separation of concerns. The placement of script tags matters: scripts placed in the head block page rendering until they load and execute (render-blocking), while scripts placed at the end of the body run after the HTML has been parsed. The modern best practice is to use the defer attribute (which loads the script in parallel but executes it after HTML parsing is complete) or the async attribute (which loads and executes the script as soon as possible, regardless of HTML parsing). Understanding this loading behavior is crucial for optimizing page performance.

**3. JavaScript's Evolution -- From ES5 to Modern JavaScript:**

JavaScript has undergone a remarkable evolution. ES5 (2009) was the stable baseline for years, supported by virtually all browsers. ES6 (also called ES2015) was a transformative update that introduced let/const declarations, arrow functions, template literals, destructuring, classes, modules, Promises, and much more -- it essentially modernized the language. Since ES6, JavaScript has followed an annual release cycle (ES2016, ES2017, ..., through to the present), with each version adding incremental improvements like async/await (ES2017), optional chaining and nullish coalescing (ES2020), top-level await (ES2022), and more. Modern browsers support these features natively, and tools like Babel can transpile modern JavaScript to older syntax for backward compatibility. As a developer, you should write modern ES6+ JavaScript and use build tools to ensure compatibility with older browsers when needed.

**4. Why JavaScript Matters Beyond the Browser:**

JavaScript's influence extends far beyond web browsers. Node.js brought JavaScript to the server side, enabling full-stack development with a single language. Frameworks like React Native and Ionic allow building mobile apps with JavaScript. Electron powers desktop applications (VS Code, Slack, and Discord are all built with it). Libraries like TensorFlow.js bring machine learning to JavaScript. The npm ecosystem (Node Package Manager) is the largest software registry in the world, with over two million packages. Understanding JavaScript opens doors not just to frontend web development, but to an entire ecosystem of tools, platforms, and career opportunities.`,
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
					Content: `Variables and data types form the absolute foundation of JavaScript programming. Variables are named containers that store values in your program's memory, and data types define what kind of values those containers can hold. Understanding these concepts deeply is crucial because every operation in JavaScript -- from simple arithmetic to complex DOM manipulation -- involves storing, retrieving, and transforming data through variables. Getting the fundamentals right here will prevent countless bugs and frustrations as you build more complex programs.

**1. Variable Declarations -- let, const, and the Legacy of var:**

JavaScript provides three keywords for declaring variables, each with different scoping and reassignment rules. The var keyword is the original declaration method from early JavaScript. It is function-scoped (meaning it is visible throughout the entire function it is declared in, regardless of block boundaries like if statements or loops) and can be redeclared and reassigned freely. However, var has a quirk called "hoisting" -- variable declarations are moved to the top of their scope during compilation, which can lead to confusing bugs where a variable appears to exist before the line where you declared it. For these reasons, var is considered legacy and should be avoided in modern code. The let keyword (introduced in ES6) is block-scoped, meaning it is only accessible within the curly braces where it is declared. It can be reassigned but not redeclared in the same scope, making it safer and more predictable than var. The const keyword is also block-scoped but adds an additional constraint: the variable cannot be reassigned after initialization. This does not mean the value is immutable -- if you assign an object or array to a const, you can still modify its contents (add properties, push items), but you cannot reassign the variable to point to a different value entirely. Best practice is to use const by default and only switch to let when you know a variable needs to be reassigned.

**2. Data Types -- The Building Blocks of JavaScript Values:**

JavaScript has seven primitive types and one complex type. Strings represent text and can be defined with single quotes, double quotes, or template literal backticks. Numbers represent both integers and floating-point values -- JavaScript does not distinguish between them, using 64-bit floating-point (IEEE 754) for all numbers, which occasionally causes precision issues (the famous 0.1 + 0.2 !== 0.3 problem). Booleans are simple true/false values used in conditional logic. Null explicitly represents "no value" or "empty" -- it is an intentional absence of data. Undefined means a variable has been declared but not yet assigned a value. Symbol (ES6) creates unique, immutable identifiers, primarily used as object property keys to avoid name collisions. BigInt (ES2020) handles integers larger than Number.MAX_SAFE_INTEGER (2^53 - 1), necessary for precise large-number arithmetic. Beyond primitives, everything else in JavaScript is an object -- this includes plain objects (key-value collections), arrays (ordered lists), functions (which are first-class objects in JavaScript), dates, regular expressions, and more.

**3. Type Coercion -- JavaScript's Double-Edged Sword:**

One of JavaScript's most notorious features is implicit type coercion -- the language automatically converts values between types when performing operations with mixed types. For example, adding a string and a number concatenates them as strings ("5" + 3 becomes "53"), while subtracting converts the string to a number ("5" - 3 becomes 2). This behavior can lead to subtle, hard-to-find bugs. The equality operator (==) performs type coercion before comparison, so "5" == 5 is true. The strict equality operator (===) does not perform coercion, so "5" === 5 is false. Best practice is to always use === and !== to avoid coercion surprises. Understanding "truthy" and "falsy" values is also essential: values like 0, empty string (""), null, undefined, NaN, and false are falsy (they evaluate to false in boolean contexts), while everything else is truthy. This affects if statements, ternary operators, and logical operators in ways that can be surprising to newcomers.

**4. Type Checking and Template Literals:**

The typeof operator returns a string indicating the type of a value (typeof "hello" returns "string", typeof 42 returns "number"). However, it has a famous quirk: typeof null returns "object" instead of "null" -- this is a bug from JavaScript's earliest implementation that has been preserved for backward compatibility. For checking arrays, use Array.isArray() instead of typeof (which returns "object" for arrays). The instanceof operator checks whether an object is an instance of a particular constructor. Template literals (strings enclosed in backtick characters) are one of ES6's most useful additions, allowing embedded expressions via ${} syntax and multi-line strings without escape characters -- they make string construction far more readable and less error-prone than traditional concatenation with the + operator.`,
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
					Content: `Functions and control flow are the two pillars that transform JavaScript from a simple calculator into a powerful programming language capable of expressing complex logic. Functions allow you to encapsulate reusable pieces of logic, giving them names and making your code modular, testable, and maintainable. Control flow structures let you make decisions (branching) and repeat operations (looping), enabling your programs to respond dynamically to different data and conditions. Together, they are the fundamental building blocks of every JavaScript application, from a simple form validator to a complex single-page application.

**1. Function Types -- Multiple Ways to Define Reusable Logic:**

JavaScript offers several ways to define functions, each with subtle differences. Function declarations use the function keyword followed by a name (function add(a, b) { return a + b; }). They are hoisted, meaning you can call them before the line where they are defined in your code -- the JavaScript engine moves the declaration to the top of the scope during compilation. Function expressions assign an anonymous or named function to a variable (const add = function(a, b) { return a + b; }). They are not hoisted, so you must define them before calling them. Arrow functions (introduced in ES6) provide a concise syntax (const add = (a, b) => a + b) and, crucially, do not have their own "this" binding -- they inherit "this" from their surrounding scope, which eliminates a common source of bugs in event handlers and callbacks. For single-expression arrow functions, the return is implicit (no need for curly braces or the return keyword). Immediately Invoked Function Expressions (IIFEs) are functions that execute immediately upon definition, historically used to create private scopes, though modern block scoping with let/const has made this pattern less necessary.

**2. Control Flow -- Making Decisions and Repeating Actions:**

Control flow structures determine which code runs and how many times. The if/else statement is the most fundamental branching mechanism -- it evaluates a condition and executes one block of code if true, optionally another if false, and can chain multiple conditions with else if. The switch statement is an alternative for comparing a single value against multiple possible matches, often cleaner than long if/else chains when checking many discrete values. The ternary operator (condition ? valueIfTrue : valueIfFalse) is a concise inline conditional, perfect for simple either/or assignments but best avoided for complex logic. For loops come in several flavors: the classic for loop (for (let i = 0; i < array.length; i++)) gives you complete control over initialization, condition, and increment. The for...of loop (for (const item of array)) iterates over iterable values (arrays, strings, Maps, Sets) and is the most readable way to loop through collections. The for...in loop iterates over object property keys (use with caution, as it also iterates inherited properties). The while loop repeats as long as a condition is true, and the do...while loop guarantees at least one execution before checking the condition.

**3. Function Parameters -- Flexibility and Expressiveness:**

Modern JavaScript provides powerful parameter features that make functions more flexible. Default parameters (function greet(name = "Guest")) let you specify fallback values when arguments are not provided, eliminating the need for manual checks inside the function body. Rest parameters (function sum(...numbers)) collect all remaining arguments into an array, enabling functions that accept any number of arguments. This is far cleaner than the old arguments pseudo-array object. Destructuring in parameters lets you extract specific properties from objects passed as arguments (function greet({ name, age })), making function signatures self-documenting and callers flexible about argument order.

**4. Scope -- Where Variables Live and Die:**

Scope determines where in your code a variable can be accessed. Global scope variables are accessible everywhere, which sounds convenient but creates risks of naming conflicts and unintended modifications -- minimize global variables in professional code. Function scope (created by function definitions) means variables declared with var inside a function are accessible throughout that function but not outside it. Block scope (created by curly braces in if statements, loops, etc.) applies to variables declared with let and const -- they are accessible only within the block where they are declared. Understanding scope is essential for avoiding bugs: closures (functions that "remember" variables from their enclosing scope even after that scope has finished executing) are one of JavaScript's most powerful patterns, enabling data privacy, factory functions, and callback patterns that are fundamental to the language.`,
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
					Content: `Objects and arrays are the two most important data structures in JavaScript, and you will use them constantly in virtually every program you write. Objects are unordered collections of key-value pairs, perfect for representing entities with named properties -- like a user with a name, age, and email, or a product with a title, price, and description. Arrays are ordered, indexed lists of values, ideal for representing collections of similar items -- like a list of usernames, a series of temperatures, or a set of search results. Understanding how to create, access, modify, and transform objects and arrays is essential for working with APIs, managing application state, rendering dynamic user interfaces, and processing data of any kind.

**1. Objects -- Modeling Real-World Entities:**

Objects in JavaScript are created most commonly using object literal syntax -- curly braces containing comma-separated key-value pairs. Keys (also called property names) are strings (or Symbols), and values can be any data type, including other objects, arrays, and functions. When a function is stored as an object property, it is called a method. You access properties using dot notation (person.name) for simple, known property names, or bracket notation (person["first name"]) when the property name contains special characters, is stored in a variable, or is computed dynamically. JavaScript provides several built-in methods for working with objects: Object.keys() returns an array of property names, Object.values() returns an array of property values, and Object.entries() returns an array of [key, value] pairs -- these are invaluable for iterating over object contents. Destructuring (const { name, age } = person) lets you extract multiple properties into individual variables in a single, elegant statement, with support for renaming (const { name: userName } = person) and default values (const { role = "user" } = person).

**2. Arrays -- Working with Ordered Collections:**

Arrays are zero-indexed, meaning the first element is at position 0, the second at position 1, and so on. They are dynamically sized -- you can add or remove elements at any time without declaring a fixed size. What makes JavaScript arrays truly powerful is their rich set of built-in methods for transforming and querying data. The map() method creates a new array by applying a function to every element (like doubling every number or extracting a specific property from every object). The filter() method creates a new array containing only elements that pass a test (like finding all users older than 18 or all products under a certain price). The reduce() method "reduces" an entire array to a single value by accumulating results (like summing all numbers, or building a summary object from a list). The forEach() method simply executes a function for each element without returning anything. The find() method returns the first element matching a condition, while findIndex() returns its position. The some() and every() methods test whether any or all elements pass a condition. Learning to chain these methods (array.filter(...).map(...).sort(...)) is a hallmark of fluent JavaScript programming and enables elegant, readable data transformations.

**3. Array and Object Destructuring -- Elegant Data Extraction:**

Destructuring is one of ES6's most transformative features, allowing you to "unpack" values from arrays and properties from objects into distinct variables with concise, readable syntax. Array destructuring uses position-based assignment (const [first, second, ...rest] = myArray), letting you grab specific elements and collect the remainder with the rest operator. You can skip elements with empty slots (const [, , third] = myArray). Object destructuring uses property name matching (const { name, age } = person), and you can rename variables (const { name: fullName } = person), set defaults for missing properties, and destructure nested objects in a single statement. Destructuring is used everywhere in modern JavaScript: in function parameters, in import statements, in loop iterations, and in React component props. The spread operator (...) complements destructuring by letting you copy and merge arrays ([...arr1, ...arr2]) and objects ({...obj1, ...obj2}) immutably -- creating new collections without modifying the originals.

**4. Common Data Manipulation Patterns:**

Real-world programming constantly involves transforming data between different shapes. A common pattern is converting an array of objects into a lookup map (using reduce() to create an object keyed by ID). Another is flattening nested arrays (using flat() or flatMap()). Sorting arrays requires understanding that sort() modifies the original array and uses lexicographic string comparison by default -- to sort numbers correctly, you must provide a comparison function (array.sort((a, b) => a - b)). For immutable operations (critical in frameworks like React where state should not be mutated directly), always create new arrays and objects using spread syntax rather than modifying existing ones. Deep cloning of nested objects requires special handling -- structuredClone() (available in modern environments) creates a true deep copy, unlike spread syntax which only performs shallow copying.`,
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
					Content: `Error handling is what separates amateur scripts that crash unpredictably from professional applications that degrade gracefully and communicate clearly with users when something goes wrong. In the real world, things fail constantly: network requests time out, users enter invalid data, APIs return unexpected responses, files go missing, and third-party services go down. Without proper error handling, a single failure can crash your entire application, leaving users staring at a blank screen or a cryptic error message. Well-implemented error handling catches failures, provides meaningful feedback to users, logs technical details for developers, and ensures the application continues functioning as well as possible. Think of error handling as the safety net under a trapeze act -- it does not prevent falls, but it prevents catastrophe when they happen.

**1. Built-in Error Types -- Understanding What Went Wrong:**

JavaScript provides several built-in error types, each indicating a specific category of problem. SyntaxError occurs when JavaScript cannot parse your code due to invalid syntax -- like a missing parenthesis or an unexpected token. These are typically caught during development rather than at runtime. ReferenceError is thrown when you try to access a variable that has not been declared -- often caused by typos in variable names or accessing variables outside their scope. TypeError is one of the most common runtime errors, thrown when you perform an operation on the wrong type -- like calling a method on undefined (the infamous "Cannot read property of undefined" error) or trying to invoke something that is not a function. RangeError occurs when a numeric value is outside the allowable range, like creating an array with a negative length or calling a function with too much recursion. URIError occurs with malformed URI encoding/decoding. These built-in types help you quickly categorize and respond to different failure modes.

**2. Try-Catch-Finally -- The Error Handling Mechanism:**

The try-catch-finally statement is JavaScript's primary error handling mechanism. Code that might throw an error goes inside the try block. If an error occurs, execution immediately jumps to the catch block, where you receive the error object and can inspect its message, name, and stack trace to understand what went wrong. The finally block (optional) executes regardless of whether an error occurred -- it is perfect for cleanup operations like closing connections, clearing timers, or resetting UI state. The throw keyword lets you create and throw your own errors when your code detects an invalid condition (throw new Error("Email address is required")). You can also create custom error classes by extending the built-in Error class, adding properties specific to your application domain (like a status code, an error code, or a list of validation failures). This allows catch blocks to identify error types using instanceof and respond differently to different failure categories.

**3. Error Handling in Asynchronous Code:**

Asynchronous operations introduce additional error handling considerations. With Promises, errors are caught using the .catch() method chained to the promise, or they propagate to the nearest .catch() in a promise chain. Unhandled promise rejections (promises that fail without a .catch()) used to silently disappear, but modern JavaScript environments (browsers and Node.js) now emit warnings or even terminate the process for unhandled rejections. With async/await syntax, you wrap await calls in try-catch blocks, which feels much more natural and readable than chaining .then() and .catch(). For fetch() API calls, remember that network errors (like no internet connection) throw exceptions, but HTTP error responses (4xx and 5xx status codes) do not -- you must manually check response.ok and throw an error if the response indicates failure. This is a common gotcha that causes silent failures in many applications.

**4. Best Practices for Professional Error Handling:**

Only catch errors you can meaningfully handle -- catching an error just to ignore it masks problems and makes debugging nearly impossible. Provide user-friendly error messages in the UI (like "Unable to save your changes. Please try again.") while logging detailed technical information (stack traces, request data, error codes) to your error tracking service. Use centralized error handling where possible -- in React, this means error boundaries; in Express.js, this means error middleware; in vanilla JavaScript, this means global error event listeners. Validate input data before processing it to prevent errors from occurring in the first place -- it is always better to reject invalid data at the boundary than to handle the errors it causes deep inside your application logic. Consider implementing retry logic with exponential backoff for transient failures like network timeouts. Never expose stack traces or internal error details to end users, as they can reveal sensitive implementation details and security vulnerabilities.`,
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
			ID:          1503,
			Title:       "DOM Manipulation",
			Description: "Learn to interact with HTML elements using JavaScript: selecting, modifying, and event handling.",
			Order:       3,
			Lessons: []problems.Lesson{
				{
					Title: "Understanding the DOM",
					Content: `The Document Object Model (DOM) is the critical bridge between your static HTML document and the dynamic, interactive web applications users expect. When a browser loads an HTML page, it does not simply display the raw text -- it parses the HTML and constructs an in-memory tree structure called the DOM, where every element, attribute, and piece of text becomes a programmable "node" object. JavaScript interacts with this tree to read content, modify elements, add new elements, remove existing ones, change styles, and respond to user actions. Without the DOM, JavaScript would have no way to "see" or "touch" the web page. Understanding the DOM is the key that unlocks the transition from writing static HTML pages to building interactive web applications.

**1. The DOM as a Tree -- Visualizing the Structure:**

The DOM represents your HTML document as a hierarchical tree, much like a family tree or an organizational chart. At the very top is the Document node, which represents the entire page. Its first child is typically the html element, which in turn has two children: head and body. Each element in your HTML becomes an Element node in the DOM tree, with its child elements becoming child nodes, creating a parent-child-sibling structure that mirrors your HTML nesting. But the DOM tree contains more than just elements -- text inside elements becomes Text nodes, comments become Comment nodes, and attributes are accessible as properties on Element nodes. Understanding this tree structure is fundamental because every DOM operation -- selecting elements, traversing relationships, inserting content -- works within this hierarchical model. When you call a method like querySelector, you are essentially searching through this tree to find a matching node.

**2. Selecting Elements -- Finding Nodes in the Tree:**

Before you can modify anything in the DOM, you must first select the element(s) you want to work with. JavaScript provides several methods for this, each with different use cases and performance characteristics. getElementById() is the fastest and most specific -- it finds a single element by its unique id attribute. getElementsByClassName() and getElementsByTagName() return live HTMLCollections (lists that automatically update when the DOM changes) of all elements matching a class name or tag name. However, the modern workhorses of DOM selection are querySelector() and querySelectorAll(). querySelector() accepts any valid CSS selector string and returns the first matching element -- you can use class selectors (".myClass"), ID selectors ("#myId"), attribute selectors ("[data-type='button']"), combinators ("nav > ul > li"), and even pseudo-classes. querySelectorAll() returns a static NodeList of all matching elements. These methods are preferred in modern development because they use the same selector syntax as CSS, making them intuitive and powerful. The key distinction between live collections (getElementsBy*) and static node lists (querySelectorAll) matters: live collections reflect subsequent DOM changes, while static lists are snapshots frozen at the time of the query.

**3. Modifying Content and Attributes:**

Once you have selected an element, you can modify its content, attributes, and appearance. The innerHTML property gets or sets the HTML markup inside an element -- it is powerful but carries security risks if you insert unsanitized user input (XSS attacks). The textContent property gets or sets the raw text content, stripping all HTML tags -- it is safer and more performant than innerHTML for plain text. The innerText property is similar to textContent but is aware of CSS styling (it returns only visible text, respecting display: none and other visibility styles). For attributes, setAttribute(), getAttribute(), and removeAttribute() provide complete control, while many common attributes (src, href, value, checked) can be accessed directly as element properties. The dataset property provides convenient access to data-* attributes (element.dataset.userId reads data-user-id).

**4. Styling and CSS Classes:**

The style property provides direct access to an element's inline CSS styles -- element.style.color = "red" sets the color, element.style.backgroundColor = "blue" sets the background (note the camelCase property names in JavaScript versus the kebab-case in CSS). While useful for dynamic, JavaScript-driven style changes, manipulating inline styles directly should be used sparingly -- it is generally better to add or remove CSS classes that are defined in your stylesheet. The classList property provides an elegant API for this: classList.add() adds a class, classList.remove() removes one, classList.toggle() flips a class on or off, and classList.contains() checks if a class is present. This approach keeps your styling in CSS where it belongs while using JavaScript only to trigger state changes.`,
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
					Content: `Events are the heartbeat of interactive web applications. Every time a user clicks a button, types in a text field, scrolls the page, hovers over a menu item, submits a form, or even just moves their mouse, the browser fires an event -- a signal that something has happened. JavaScript's event system allows you to "listen" for these signals and execute code in response, creating the illusion of a responsive, living interface. Without events, web pages would be completely static -- you could read content but never interact with it. Understanding events deeply is not optional for a frontend developer; it is the core skill that enables every interactive feature you build.

**1. The Event System -- How Browsers Communicate User Actions:**

The browser constantly monitors user interactions and system occurrences, converting them into event objects that your JavaScript code can receive and process. Mouse events (click, dblclick, mousedown, mouseup, mousemove, mouseover, mouseout, mouseenter, mouseleave) track pointer interactions. Keyboard events (keydown, keyup, keypress) capture typing. Form events (submit, change, input, focus, blur) monitor form control interactions. Window events (load, resize, scroll, unload) respond to browser-level changes. Touch events (touchstart, touchmove, touchend) handle mobile interactions. There are dozens of event types, each designed for a specific category of interaction. Understanding which event to listen for is the first step to building any interactive feature.

**2. addEventListener -- The Modern Event Handling Standard:**

The addEventListener() method is the correct, modern way to attach event handlers to DOM elements. It accepts the event type as a string (like "click" or "submit"), a callback function that executes when the event fires, and an optional options object. Its key advantages over older approaches (inline onclick attributes and on-event properties) are numerous: you can attach multiple handlers for the same event on the same element, you can remove specific handlers later with removeEventListener(), and you have fine-grained control over event propagation phases. The callback function automatically receives an event object as its first argument, containing detailed information about what happened. Inline event handlers (onclick="doSomething()") mix JavaScript into HTML, violating separation of concerns, and on-event properties (element.onclick = handler) only allow one handler per event per element, silently overwriting any previous handler. Always use addEventListener in production code.

**3. The Event Object -- Rich Context About What Happened:**

When an event fires, the browser creates an event object packed with useful information and methods. The event.target property references the actual element that triggered the event (for example, the specific list item clicked, even if the listener is on the parent list). The event.currentTarget property references the element the listener is attached to. The event.type tells you what kind of event occurred. The event.preventDefault() method stops the browser's default behavior for that event -- essential for intercepting form submissions (to validate before sending), preventing link navigation (for single-page application routing), or stopping right-click context menus. The event.stopPropagation() method prevents the event from continuing to bubble up through the DOM tree. For keyboard events, event.key tells you which key was pressed in a human-readable format ("Enter", "Escape", "a"), while event.ctrlKey, event.shiftKey, and event.altKey indicate whether modifier keys were held. For mouse events, event.clientX and event.clientY give you the cursor coordinates relative to the viewport.

**4. Event Delegation -- Efficient Handling of Dynamic Content:**

Event delegation is a powerful pattern that takes advantage of event bubbling -- the fact that events fired on a child element automatically propagate ("bubble") up through every ancestor element in the DOM tree. Instead of attaching individual event listeners to every child element (which is wasteful and breaks when new children are added dynamically), you attach a single listener to a common parent and use event.target to determine which child actually triggered the event. For example, instead of adding a click handler to each of 100 list items, you add one handler to the list element itself. When any list item is clicked, the event bubbles up to the list, where your handler inspects event.target to identify which item was clicked. This pattern is essential for dynamic applications where elements are frequently added, removed, or replaced -- the parent listener automatically covers new elements without any additional setup. It also significantly reduces memory usage since you maintain one listener instead of hundreds.`,
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
					Content: `The ability to dynamically create, insert, and remove DOM elements is what transforms static web pages into living, breathing applications. Every time you see a new notification badge appear, a chat message slide into view, a shopping cart item get added, a to-do list entry get checked off and removed, or an infinite scroll page load more content, the underlying mechanism is DOM manipulation -- JavaScript creating new elements, inserting them into the page, and removing elements that are no longer needed. This is the foundation of every modern web application, and before frameworks like React or Vue abstracted this away, every interactive feature was built through direct DOM manipulation.

**1. Creating Elements -- Building New DOM Nodes Programmatically:**

The document.createElement() method is the primary tool for creating new elements. You pass it a tag name (like "div", "p", "button", or "img") and it returns a brand-new Element node that exists in memory but is not yet part of the visible page. You can then configure this element -- set its textContent, add CSS classes via classList, set attributes with setAttribute(), attach event listeners with addEventListener() -- all before it ever appears on screen. The createTextNode() method creates a standalone text node that you can append inside an element. The cloneNode() method creates a copy of an existing element; passing true as an argument creates a "deep clone" that includes all descendant elements and text, while false (or no argument) creates a "shallow clone" of just the element itself. While innerHTML can also create elements by parsing an HTML string, this approach is less secure because inserting unsanitized user input via innerHTML opens your application to Cross-Site Scripting (XSS) attacks -- a malicious user could inject script tags that execute arbitrary code.

**2. Inserting Elements -- Placing New Nodes in the Document:**

Once you have created and configured an element, you need to insert it into the DOM tree for it to become visible. The appendChild() method adds a node as the last child of a parent element -- this is the most commonly used insertion method. The prepend() method adds a node as the first child. The insertBefore() method inserts a node before a specified reference child, giving you precise control over placement. The insertAdjacentHTML() method is uniquely flexible: it accepts a position string ("beforebegin", "afterbegin", "beforeend", or "afterend") and an HTML string, letting you insert content at four different positions relative to the target element. For inserting plain text safely, insertAdjacentText() works similarly but without HTML parsing. The after() and before() methods (on the reference element itself) provide a modern, readable API for sibling insertion. Understanding the performance implications of DOM insertion is important: every time you insert or remove an element, the browser must recalculate layout and potentially repaint the screen. For inserting many elements at once, use a DocumentFragment (a lightweight, invisible container) to batch your insertions and trigger only a single reflow.

**3. Removing Elements -- Cleaning Up the DOM:**

Removing elements is just as important as creating them. The modern remove() method, called directly on the element (element.remove()), is the simplest approach. The older removeChild() method is called on the parent element (parent.removeChild(child)) and returns the removed node, which can be useful if you want to reinsert it elsewhere later. Setting innerHTML to an empty string (container.innerHTML = "") clears all children at once, which is efficient for complete content replacements but should be used carefully as it also removes all event listeners on the cleared elements. The replaceChild() method (or the modern replaceWith()) swaps one element for another in a single operation, useful for live-updating content like notification counts or status indicators.

**4. Best Practices for DOM Manipulation:**

Prefer createElement() over innerHTML for security -- never insert unsanitized user input as HTML. When creating many elements, build them in a DocumentFragment or off-screen container, then insert the entire batch at once to minimize expensive browser reflows and repaints. Clean up event listeners when removing elements to prevent memory leaks -- if you added a listener to an element, remove it before removing the element from the DOM (though garbage collection handles this in most modern browsers when using addEventListener without external references). Consider using the template element for complex structures -- define the HTML template once and clone it for each instance, which is more maintainable than building complex DOM trees purely in JavaScript. In modern development, frameworks like React, Vue, and Angular abstract direct DOM manipulation behind a virtual DOM or reactive data binding, but understanding the raw DOM API remains invaluable for debugging, performance optimization, and working with third-party libraries.`,
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
					Content: `DOM traversal is the art of navigating through the DOM tree -- moving from one node to its parents, children, and siblings to find related elements. Think of it like navigating a family tree: starting from any person, you can move up to their parents, down to their children, or sideways to their siblings. This capability is essential in interactive applications where an event on one element (like a click on a "delete" button) needs to affect a related element (like removing the parent list item that contains the button). While querySelector is often the most direct way to find a specific element, traversal methods are indispensable when you need to work with elements relative to a known starting point, particularly in event handlers and component logic.

**1. Parent Navigation -- Moving Up the Tree:**

Every element in the DOM (except the root Document node) has a parent. The parentNode property returns the direct parent node, which is usually an Element but could be the Document node itself for the top-level html element. The parentElement property is similar but returns null if the parent is not an Element node (this distinction rarely matters in practice). The real powerhouse for upward navigation is the closest() method, which searches up through an element's ancestors (including the element itself) and returns the first one matching a given CSS selector. This is extraordinarily useful in event handling: imagine clicking a "remove" button deep inside a card component -- event.target.closest(".card") immediately gives you the containing card element, regardless of how deeply nested the button is. The closest() method makes event delegation patterns much cleaner and more robust, because you do not need to make assumptions about how many levels of nesting separate the clicked element from the container you want to operate on.

**2. Child Navigation -- Moving Down the Tree:**

Navigating to an element's children is essential for iterating over lists, processing table rows, or working with any container's contents. The children property returns an HTMLCollection of only Element children (ignoring text nodes, comments, and whitespace), which is almost always what you want. The childNodes property returns a NodeList of all child nodes including text and comments, which is useful for lower-level DOM processing but cluttered for most practical purposes. firstElementChild and lastElementChild give direct access to the first and last child elements respectively, while firstChild and lastChild return any type of node (often a whitespace text node, which can be surprising). To iterate over children, you can convert the HTMLCollection to an array with Array.from(element.children) or the spread syntax [...element.children], then use standard array methods like forEach, map, and filter. The childElementCount property tells you how many element children exist without needing to access the full collection.

**3. Sibling Navigation -- Moving Sideways in the Tree:**

Sibling navigation lets you move to adjacent elements at the same level of the tree. The nextElementSibling and previousElementSibling properties return the next and previous Element siblings, skipping over text and comment nodes. Their counterparts, nextSibling and previousSibling, return any type of node and are rarely needed in practice. Sibling navigation is particularly useful for building keyboard-navigable widgets: when the user presses the down arrow key on a menu item, you can focus the nextElementSibling; when they press up, focus the previousElementSibling. It is also useful for reordering elements, toggling active states, and implementing before/after insertion logic. Always check for null when navigating to siblings, as the first element has no previousElementSibling and the last has no nextElementSibling.

**4. Combining Traversal with Query Methods:**

While traversal properties navigate one step at a time through the tree, query methods let you search deeply within a subtree. You can call querySelector() and querySelectorAll() on any element (not just document), restricting the search to that element's descendants. The matches() method checks whether a given element matches a CSS selector, which is useful for filtering during traversal. For performance, cache DOM references in variables rather than repeatedly querying the DOM -- each query involves searching the tree, and excessive queries in tight loops (like animations or scroll handlers) can degrade performance. When writing traversal code, always implement null checks (if (element.parentElement) {...}) to prevent errors when you reach the edges of the DOM tree. Consider using optional chaining (element?.closest(".card")?.querySelector(".title")) for elegant, safe traversal chains in modern JavaScript.`,
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
			ID:          1504,
			Title:       "Responsive Design",
			Description: "Create websites that work on all devices: mobile-first approach, media queries, and flexible layouts.",
			Order:       4,
			Lessons: []problems.Lesson{
				{
					Title: "Introduction to Responsive Design",
					Content: `Responsive web design is the practice of building websites that automatically adapt their layout, content, and visual presentation to look great and function well on any device, from a small smartphone screen to a large desktop monitor and everything in between. In today's world, where over 60% of web traffic comes from mobile devices, responsive design is not a nice-to-have feature -- it is an absolute requirement. Google uses mobile-friendliness as a ranking factor, and users who encounter a website that does not work well on their device will leave within seconds. The term "responsive web design" was coined by Ethan Marcotte in 2010, and it rests on three foundational pillars: fluid layouts, flexible media, and CSS media queries.

**1. The Philosophy Behind Responsive Design:**

Before responsive design, companies often maintained separate websites for desktop and mobile (like m.example.com), which meant duplicate content, duplicate maintenance, and a fragmented user experience. Responsive design eliminates this by using a single HTML codebase that adapts its presentation through CSS. Think of it like water: just as water takes the shape of whatever container it is poured into, a responsive website takes the shape of whatever screen it is viewed on. The layout rearranges, font sizes adjust, images resize, and navigation transforms -- all automatically, without requiring the user to do anything. This approach saves development time, ensures content consistency, improves SEO (Google prefers a single URL per page), and provides a seamless experience for users who switch between devices throughout the day.

**2. Fluid Layouts -- The Foundation of Responsiveness:**

The first key to responsive design is using fluid, percentage-based widths instead of fixed pixel values. A container set to width: 80% will take up 80% of its parent's width regardless of screen size, naturally shrinking on smaller screens and expanding on larger ones. The max-width property is a crucial companion: setting max-width: 1200px on a container that has width: 100% means it fills the screen on small devices but caps at 1200px on large monitors, preventing excessively long line lengths that harm readability. This combination of percentage widths and max-width constraints is the fundamental pattern for fluid containers. CSS Grid's fr unit and Flexbox's flex-grow property take fluid layouts further, distributing available space proportionally among child elements.

**3. The Viewport Meta Tag -- The Essential Configuration:**

The viewport meta tag is a single line of HTML that is absolutely critical for responsive design on mobile devices. Without it, mobile browsers render pages at a default virtual viewport width (typically 980px) and then shrink the result to fit the screen, producing a tiny, zoomed-out view of a desktop layout. Adding the tag with content="width=device-width, initial-scale=1.0" tells the browser to set the viewport width equal to the device's actual screen width and start at a 1:1 zoom level. This single tag enables your media queries and fluid layouts to work correctly on mobile devices. Every responsive website must include it -- forgetting it is one of the most common reasons a "responsive" site still looks broken on phones.

**4. Breakpoints and Responsive Units:**

Breakpoints are the viewport widths at which your layout changes significantly. Common breakpoints align with typical device categories: mobile (below 768px), tablet (768px to 1024px), and desktop (above 1024px), but you should choose breakpoints based on where your specific content breaks rather than targeting specific devices -- devices come in an ever-growing variety of sizes. Responsive units are essential tools: percentages for container widths, rem for typography (respects user font-size preferences and scales consistently), em for component-relative spacing, and viewport units (vw, vh) for viewport-relative sizing. The modern clamp() function (like font-size: clamp(1rem, 2.5vw, 2rem)) creates smoothly scaling values without breakpoints. The min() and max() functions provide similar flexibility for other properties.`,
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
					Content: `Media queries are the conditional logic of CSS -- they allow you to apply different styles based on the characteristics of the user's device, viewport, or preferences. Just as an if/else statement in JavaScript lets your program behave differently based on conditions, media queries let your stylesheet adapt based on screen width, orientation, resolution, color scheme preference, and more. They are the mechanism that makes responsive design possible, enabling a single stylesheet to produce dramatically different layouts for phones, tablets, and desktops. Media queries were introduced in CSS3 and have become one of the most important CSS features for modern web development.

**1. Media Query Syntax -- The Structure of Conditional CSS:**

A media query consists of the @media rule, an optional media type, and one or more conditions (called "media features") enclosed in parentheses. The styles inside the media query's curly braces only apply when the conditions are met. For example, @media screen and (min-width: 768px) { ... } applies its styles only on screen devices with a viewport at least 768 pixels wide. The media type (screen, print, speech, all) specifies which class of device the styles target -- "screen" covers computers, tablets, and phones; "print" applies when the page is printed; "speech" targets screen readers; and "all" (the default) applies everywhere. In practice, most developers omit the media type entirely, since "all" is the default and most responsive styles are relevant across all media types.

**2. Media Features -- What You Can Query:**

Width-based features are by far the most commonly used. min-width matches when the viewport is at least a specified width, making it the foundation of mobile-first design (your base styles are for mobile, and min-width queries progressively add styles for larger screens). max-width matches when the viewport is at most a specified width, used in desktop-first approaches. You can combine both to create range conditions: @media (min-width: 768px) and (max-width: 1023px) targets only tablet-sized screens. The orientation feature matches portrait or landscape mode, useful for adjusting layouts when users rotate their devices. The resolution feature (or min-resolution/max-resolution) matches based on screen pixel density, useful for serving higher-resolution images to Retina displays. Modern media queries also support user preference features: prefers-color-scheme matches "light" or "dark" based on the user's OS theme setting, prefers-reduced-motion matches users who have requested reduced animations (important for accessibility), and prefers-contrast matches users who want high or low contrast.

**3. Logical Operators -- Combining Conditions:**

Media queries support logical operators that let you build complex conditions. The "and" keyword requires all conditions to be true: @media (min-width: 768px) and (orientation: landscape) matches only landscape viewports at least 768px wide. The comma operator acts as logical OR: @media (max-width: 767px), (orientation: portrait) matches viewports that are either narrow OR in portrait orientation (or both). The "not" keyword negates an entire media query: @media not print { ... } applies to everything except printers. The "only" keyword was historically used to prevent older browsers (that do not understand media queries) from applying the styles, though it is rarely needed in modern development. These operators enable precise targeting that adapts not just to screen sizes but to combinations of device characteristics and user preferences.

**4. Best Practices for Effective Media Queries:**

Adopt a mobile-first approach: write your base styles for the smallest screens and use min-width queries to progressively enhance for larger viewports. This ensures mobile users (often on slower connections) download only the CSS they need, while desktop users receive additional layout rules. Choose breakpoints based on where your content naturally needs to reflow, not based on specific device dimensions -- the device landscape changes constantly, but content-driven breakpoints remain relevant. Test on real devices whenever possible, as device simulators do not perfectly replicate touch interactions, browser chrome, and rendering quirks. Ensure all touch targets (buttons, links, form controls) are at least 44x44 pixels on touch devices to meet accessibility guidelines. Use relative units inside media queries when possible, and consider organizing your media queries either inline (next to the rules they modify) for component-based architecture, or grouped at the end of your stylesheet for a clear overview of all breakpoints.`,
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
					Content: `Mobile-first design and container queries represent two of the most important paradigms in modern responsive web development, each addressing a different dimension of the same fundamental challenge: how do you build interfaces that look great and function well regardless of the context in which they are viewed? Mobile-first design is a philosophy and methodology that has become the industry standard over the past decade, while container queries are a newer CSS capability that finally solves a problem web developers have been requesting for over a decade. Together, they give you a complete toolkit for building truly adaptive interfaces.

**1. The Mobile-First Approach -- Why Starting Small is Starting Smart:**

Mobile-first design means writing your base CSS styles for the smallest screens (typically smartphones around 320-375px wide) and then using min-width media queries to progressively add complexity and layout enhancements for larger screens. This might seem counterintuitive -- why not design for the big screen first and then scale down? The answer lies in both performance and design philosophy. When you start with desktop styles and then override them for mobile with max-width queries (the "desktop-first" approach), mobile users download all the desktop CSS rules only to have them immediately overridden. This wastes bandwidth on the devices that can least afford it -- mobile phones on cellular connections. With mobile-first, the base styles are lean and minimal, and additional CSS is only loaded and applied when the viewport is large enough to use it. From a design perspective, mobile-first forces you to prioritize content ruthlessly. On a small screen, there is no room for decorative sidebars, multi-column layouts, or large hero images. You must identify what truly matters to the user and present it clearly. This discipline produces cleaner, more focused designs that benefit users on every device. Think of it like packing for a trip: if you start with a small suitcase, you only bring essentials; if you start with a huge suitcase, you fill it with things you never use.

**2. Viewport Units -- Sizing Relative to the Browser Window:**

Viewport units are CSS length units that are directly proportional to the browser's visible area (the viewport), making them incredibly powerful for creating layouts that respond fluidly to the available screen real estate. The vw unit represents 1% of the viewport's width, so 100vw is the full width of the browser window. The vh unit represents 1% of the viewport's height, so 100vh is the full height of the visible area -- perfect for creating full-screen hero sections, landing pages, or modal overlays that exactly fill the screen. The vmin unit equals 1% of whichever dimension (width or height) is smaller, while vmax equals 1% of whichever dimension is larger -- these are useful for maintaining proportional sizing regardless of device orientation. Modern CSS has also introduced the "dynamic" viewport units (dvw, dvh) and "small"/"large" variants (svh, lvh) that account for the appearance and disappearance of mobile browser UI elements (like the address bar that slides away when you scroll on a phone). Viewport units are particularly powerful for responsive typography: setting font-size to something like calc(1rem + 1vw) creates text that scales smoothly with the viewport width, though the modern clamp() function (e.g., font-size: clamp(1rem, 2.5vw, 2.5rem)) is preferred because it sets both a minimum and maximum size, preventing text from becoming too tiny on small screens or absurdly large on ultrawide monitors.

**3. Container Queries -- The Component-Level Revolution:**

For years, media queries were the only tool for responsive design, and they have one significant limitation: they query the viewport (the browser window), not the container an element actually lives in. This creates problems in component-based architectures. Imagine a card component that appears in a wide main content area on one page and in a narrow sidebar on another -- with media queries alone, both instances respond to the same viewport width, even though their available space is completely different. The card in the sidebar needs to be styled for "narrow" even on a wide desktop screen. Container queries solve this by allowing you to style elements based on the size of their parent container rather than the viewport. You first designate a container by setting container-type: inline-size (or container-type: size for both dimensions) on the parent element. Then you write @container rules that apply styles when the container reaches specified dimensions. This means a component can adapt its layout based on the space actually available to it, regardless of the overall viewport size. Container queries are a game-changer for design systems and component libraries because they make components truly self-contained and portable -- a card component can define its own responsive behavior without knowing anything about the page layout it will be placed in.

**4. Best Practices for Modern Responsive Design:**

Always adopt mobile-first as your default methodology unless you have a compelling reason not to (such as building an application that will genuinely only be used on desktop). Test your designs on real physical devices whenever possible, because device simulators and browser resize tools do not perfectly replicate touch interactions, pixel density, browser chrome behavior, or performance characteristics. Use relative units (rem for typography, percentages and fr units for layout widths, viewport units for full-screen elements) instead of fixed pixel values, so your designs scale naturally. Consider container queries for any reusable component that might appear in different layout contexts -- sidebar cards, dashboard widgets, navigation components, and media objects are all excellent candidates. Combine media queries (for page-level layout decisions) with container queries (for component-level adaptations) to create a robust, layered responsive strategy that handles every scenario gracefully.`,
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
			ID:          1505,
			Title:       "CSS Flexbox & Grid",
			Description: "Master modern CSS layout: Flexbox for one-dimensional layouts and Grid for two-dimensional layouts.",
			Order:       5,
			Lessons: []problems.Lesson{
				{
					Title: "CSS Flexbox",
					Content: `Flexbox (Flexible Box Layout) is one of the two modern CSS layout systems that has fundamentally transformed how developers build web page layouts. Before Flexbox arrived, creating even simple layouts like centering an element both vertically and horizontally, building equal-height columns, or distributing space evenly between navigation items required hacky workarounds involving floats, table-cell displays, negative margins, or absolute positioning. Flexbox eliminates all of that by providing a powerful, intuitive system for distributing space and aligning items along a single axis -- either horizontally (in a row) or vertically (in a column). If CSS Grid is like laying out a city with streets and blocks in two dimensions, Flexbox is like arranging items on a single shelf -- you control how they spread out, bunch together, wrap to new lines, and align relative to each other.

**1. The Flex Container and Flex Items -- Parent Controls the Layout:**

Flexbox works through a parent-child relationship. The parent element becomes a "flex container" by setting display: flex (or display: inline-flex if you want the container itself to behave as an inline element). All direct children of the flex container automatically become "flex items" and begin following Flexbox layout rules instead of normal document flow. The critical concept to internalize is that the flex container's properties control how items are distributed and aligned as a group, while individual flex item properties let specific children override or customize their behavior within that group. This separation of concerns makes Flexbox layouts both powerful and maintainable. Two invisible axes govern everything: the main axis (the primary direction items flow, determined by flex-direction) and the cross axis (perpendicular to the main axis). When flex-direction is row (the default), the main axis runs horizontally and the cross axis runs vertically. When flex-direction is column, these flip. Understanding which axis is "main" and which is "cross" is the key to understanding why justify-content and align-items behave differently.

**2. Container Properties -- Orchestrating the Layout:**

The flex-direction property sets the main axis direction: row (left to right), row-reverse (right to left), column (top to bottom), or column-reverse (bottom to top). The flex-wrap property controls whether items are forced onto a single line (nowrap, the default, which can cause items to shrink or overflow) or allowed to wrap onto multiple lines (wrap). The justify-content property distributes items along the main axis: flex-start packs items to the start, flex-end packs them to the end, center centers them, space-between distributes items with equal space between them (no space at the edges), space-around gives each item equal space on both sides (resulting in half-size space at the edges), and space-evenly distributes perfectly equal space between and around all items. The align-items property aligns items along the cross axis: stretch (the default, which makes items fill the container's cross-axis dimension -- this is why flex items in a row layout become equal height by default), flex-start, flex-end, center, and baseline (which aligns items by their text baselines, useful when items have different font sizes). The gap property adds consistent spacing between items without needing margins, which is a cleaner approach that avoids the common problem of unwanted margins on the first or last item.

**3. Item Properties -- Individual Control Within the Group:**

While the container properties control the group behavior, individual flex items can customize their sizing and alignment. The flex-grow property determines how much of the remaining free space an item should absorb -- a value of 0 (the default) means the item will not grow beyond its content size, while a value of 1 means it will absorb its proportional share of free space. If one item has flex-grow: 2 and another has flex-grow: 1, the first gets twice as much extra space. The flex-shrink property works in reverse: it determines how much an item should shrink when the container is too small to fit all items at their natural sizes. A value of 0 prevents shrinking entirely (useful for fixed-width sidebars or icons that should not compress). The flex-basis property sets the initial size of an item before growing or shrinking occurs -- think of it as the "ideal" size. The shorthand flex property combines all three (flex: grow shrink basis), and the most common patterns are flex: 1 (grow to fill available space), flex: 0 0 auto (fixed size, no growing or shrinking), and flex: 1 1 300px (start at 300px, grow and shrink as needed). The align-self property lets an individual item override the container's align-items value for just that one item. The order property changes the visual order of an item without changing the HTML source order, though this should be used sparingly because it can confuse screen reader users who navigate in source order.

**4. Common Flexbox Patterns -- Solving Real-World Layout Challenges:**

Flexbox excels at solving layout problems that used to require complex CSS hacks. Centering content both horizontally and vertically is trivially achieved with display: flex; justify-content: center; align-items: center on the container -- a solution that took years of CSS evolution to reach. Navigation bars are natural Flexbox candidates: set the nav as a flex container with justify-content: space-between to push the logo to one side and navigation links to the other. Card layouts use flex-wrap: wrap with a flex-basis on each card (like flex: 1 1 300px) to create responsive grids that reflow naturally as the viewport changes size. Equal-height columns -- once one of the hardest CSS challenges -- are the default behavior of flex items in a row, since align-items: stretch is the default. The "holy grail" layout (header, footer, and a three-column body) is elegantly achieved by nesting flex containers: a column flex on the body for the vertical structure and a row flex on the middle section for the columns. Sticky footers (footers that always sit at the bottom of the viewport even when content is short) are solved by setting the body to display: flex; flex-direction: column; min-height: 100vh and giving the main content area flex: 1 so it absorbs all available vertical space.`,
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
					Content: `CSS Grid is the most powerful layout system CSS has ever offered, providing true two-dimensional control over both rows and columns simultaneously. While Flexbox excels at arranging items along a single axis (either a row or a column), Grid was designed from the ground up for creating complex, two-dimensional page layouts -- think of it as a sophisticated blueprint system where you define both the horizontal and vertical structure of your page, then place items precisely into that structure. Before Grid, creating complex layouts like magazine-style designs, dashboard interfaces, or asymmetric gallery layouts required deeply nested containers, fragile float hacks, or complex Flexbox nesting. Grid replaces all of that with a clean, declarative system where you describe what you want the layout to look like, and the browser figures out how to make it happen.

**1. Grid Concepts -- The Anatomy of a Grid Layout:**

Understanding Grid requires learning its vocabulary, because the system is built around several interrelated concepts. The grid container is the parent element with display: grid set on it -- all direct children become grid items. Grid lines are the invisible horizontal and vertical dividing lines that form the structure of the grid; they are numbered starting from 1 (not 0) at the top-left corner, and you reference these numbers when placing items. Grid tracks are the rows and columns between grid lines -- a column track is the space between two vertical lines, and a row track is the space between two horizontal lines. Grid cells are the individual intersections of one row track and one column track -- like a single cell in a spreadsheet. Grid areas are rectangular regions spanning one or more cells, which you can name and reference for intuitive item placement. Think of the grid container as a blank spreadsheet: grid-template-columns defines how many columns it has and how wide each is, grid-template-rows defines how many rows and how tall each is, and then you place items into specific cells or areas of that spreadsheet. The gap property (or its longhand forms row-gap and column-gap) adds gutters between tracks, providing consistent spacing without margins.

**2. Container Properties -- Defining the Grid Structure:**

The grid-template-columns and grid-template-rows properties are how you define the structure of your grid. You specify a space-separated list of track sizes: grid-template-columns: 200px 1fr 200px creates three columns where the first and last are fixed at 200px and the middle one takes up all remaining space. The fr (fractional) unit is unique to Grid and is the most important unit to understand -- it represents a fraction of the available free space. So 1fr 2fr 1fr creates three tracks where the middle one is twice as wide as the others. The repeat() function avoids repetitive code: repeat(3, 1fr) is equivalent to 1fr 1fr 1fr, and repeat(auto-fit, minmax(250px, 1fr)) creates a responsive grid that automatically adjusts the number of columns based on available space -- items are at least 250px wide and grow to fill extra space, wrapping to new rows as the container shrinks. The minmax() function sets both a minimum and maximum size for a track, giving you flexible yet constrained tracks. The grid-template-areas property is one of Grid's most elegant features: you define named regions using a string-based visual map that literally looks like your layout in ASCII art, making complex layouts self-documenting and easy to modify.

**3. Track Sizing -- Flexible and Powerful Measurement:**

Grid offers an exceptionally rich set of sizing options for tracks. Fixed units (px, em, rem) create rigid tracks that do not change size. The fr unit distributes free space proportionally -- if you have three columns at 1fr 2fr 1fr, the middle column gets 50% of the free space and the side columns each get 25%. The auto keyword sizes a track to fit its content, growing as needed but not taking more space than necessary. The minmax() function is particularly powerful: minmax(200px, 1fr) creates a track that is at least 200px wide but can grow to fill available space, preventing content from being crushed on small screens while still being flexible on large ones. The repeat() function with auto-fit or auto-fill creates intrinsically responsive grids: auto-fit collapses empty tracks so that items stretch to fill the container, while auto-fill preserves empty tracks, maintaining the column structure even when there are not enough items to fill every column. The fit-content() function creates a track that is as wide as its content but caps at a specified maximum, useful for sidebar-style elements that should shrink-wrap their content but not exceed a certain width.

**4. Item Placement -- Precise Control Over Where Things Go:**

Grid items can be placed automatically (the browser fills cells in order) or explicitly using line-based or area-based placement. The grid-column and grid-row shorthand properties specify which lines an item starts and ends at: grid-column: 1 / 3 places an item from line 1 to line 3 (spanning two columns). The span keyword offers a more intuitive alternative: grid-column: span 2 means "span two columns starting from wherever the item naturally falls." For the most readable placement, use grid-area with named template areas: after defining areas in grid-template-areas on the container, you simply assign grid-area: header to an item and it fills the "header" region -- no line numbers to count or remember. The justify-self and align-self properties control how an individual item aligns within its grid cell horizontally and vertically, while the container-level justify-items and align-items set defaults for all items (place-items is the shorthand for both). Grid also supports implicit tracks -- rows or columns that are created automatically when items are placed outside the explicitly defined grid -- and you can control their size with grid-auto-rows and grid-auto-columns. This is particularly useful for dynamic content where you do not know in advance how many items will be in the grid.`,
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
			ID:          1506,
			Title:       "Forms and Validation",
			Description: "Create interactive forms with proper validation, accessibility, and user experience.",
			Order:       6,
			Lessons: []problems.Lesson{
				{
					Title: "HTML Forms",
					Content: `HTML forms are the primary mechanism through which users communicate data to web applications, and they are far more ubiquitous than most beginners realize. Every time you log into a website, search for something on Google, fill out a registration page, post a comment, update your profile, enter your shipping address during checkout, or even click a "like" button, you are interacting with an HTML form. Forms are the bridge between the user and the server -- they collect input, package it into a structured format, and transmit it to a backend system for processing. Mastering HTML forms is essential for any frontend developer because forms are at the heart of nearly every interactive web application.

**1. The Form Element -- The Container That Makes Submission Possible:**

The form element serves as a wrapper around all the interactive controls that collect user input. Think of it as an envelope that groups related inputs together and, when submitted, sends their data to a specified destination. The action attribute tells the browser where to send the form data -- typically a URL on your server that handles the submission. The method attribute determines how the data travels: GET appends form data to the URL as query parameters (visible in the address bar, bookmarkable, and suitable for search forms or filters), while POST sends data in the request body (hidden from the URL, more secure for sensitive data like passwords, and capable of handling larger payloads). Without a form element wrapping your inputs, the browser has no mechanism to collect and submit their values together. The enctype attribute becomes critically important when your form includes file uploads -- you must set it to "multipart/form-data" so that file contents are properly encoded for transmission. For standard text-only forms, the default "application/x-www-form-urlencoded" encoding works fine.

**2. Form Controls -- The Individual Input Mechanisms:**

The input element is the most versatile form control in HTML, with its behavior determined entirely by the type attribute. A text input creates a simple single-line text field for general text entry. An email input looks similar but automatically validates email format and triggers an email-optimized keyboard on mobile devices. A password input masks typed characters with dots or asterisks for privacy. A number input restricts entry to numeric values and provides increment/decrement buttons, with optional min, max, and step attributes for constraining the range. A date input renders a native date picker (appearance varies by browser and operating system), eliminating the need for JavaScript date picker libraries in many cases. Checkbox inputs allow multiple selections from a group (like selecting multiple interests), while radio inputs restrict selection to exactly one option within a named group (like choosing a payment method). The file input allows users to select files from their device for upload. Beyond input, the textarea element provides multi-line text entry -- perfect for comments, messages, or any content that might span multiple lines, with rows and cols attributes controlling its default visible size. The select element creates dropdown menus with option children, and the button element triggers form submission (type="submit") or custom JavaScript actions (type="button").

**3. Labels, Fieldsets, and Form Structure -- Accessibility and Organization:**

The label element is one of the most important yet frequently overlooked form elements. It associates descriptive text with a specific form control, serving two critical purposes: it tells screen reader users what each input expects (without labels, a visually impaired user encounters unlabeled text fields with no context), and it improves usability for all users by expanding the clickable area -- clicking a label focuses or toggles its associated input. Labels can be linked to inputs either by setting the label's "for" attribute to match the input's "id" attribute, or by wrapping the input element directly inside the label element. Both approaches are valid, but the explicit for/id association is more flexible and commonly used. The fieldset element groups related form controls visually (with a border by default) and semantically, which is particularly important for radio button groups and checkbox sets where multiple controls share a common question or category. The legend element provides a title for the fieldset, appearing as text embedded in the fieldset's border. A well-structured form with proper labels, fieldsets, and logical tab order is not just a nice-to-have -- it is an accessibility requirement that ensures all users, regardless of ability, can successfully complete your forms.

**4. Form Attributes and Behavior -- Controlling Submission and Interaction:**

Several attributes on the form element and its controls fine-tune behavior. The autocomplete attribute (set to "on" or "off" on the form, or more specific values like "email", "tel", "street-address" on individual inputs) controls whether browsers offer to auto-fill fields based on previously entered data -- properly configured autocomplete dramatically improves user experience, especially on mobile devices where typing is slow. The novalidate attribute on the form element disables the browser's built-in HTML5 validation, which is useful when you want to handle all validation through JavaScript instead. Individual inputs support attributes like required (prevents submission until the field is filled), placeholder (shows hint text that disappears when the user starts typing -- but never use it as a substitute for a proper label), disabled (grays out the control and excludes its value from submitted data), readonly (allows viewing but prevents editing while still including the value in submissions), and pattern (validates against a regular expression). Understanding the difference between disabled and readonly is a common interview question and a practical consideration: disabled fields are excluded from form data, while readonly fields are included.`,
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
					Content: `Form validation is the process of checking that user-submitted data meets your application's requirements before it is processed, and it is one of the most critical aspects of building reliable, secure web applications. Without validation, users can accidentally submit empty fields, misformatted email addresses, passwords that are too short, or dates that do not make sense -- leading to server errors, corrupted data, and frustrated users. Good validation provides immediate, clear feedback that guides users toward correct input, reducing form abandonment and support requests. Think of validation as a friendly checkpoint: it catches mistakes early and helps users fix them before they cause problems downstream.

**1. HTML5 Built-In Validation -- The Browser Does the Heavy Lifting:**

HTML5 introduced a suite of built-in validation attributes that allow you to enforce many common constraints without writing a single line of JavaScript. The required attribute prevents form submission if the field is empty -- the browser will display a native error tooltip pointing to the first invalid field. The min and max attributes constrain numeric and date inputs to a specified range (for example, min="18" max="120" on an age field). The minlength and maxlength attributes control text length, ensuring passwords meet minimum length requirements or preventing excessively long entries. The pattern attribute accepts a regular expression that the input value must match, giving you precise control over format (like requiring a specific phone number pattern). The type attribute itself provides validation: an input with type="email" automatically checks for a valid email format, type="url" checks for a valid URL, and type="number" rejects non-numeric input. The beauty of HTML5 validation is that it is declarative (you describe constraints in HTML attributes rather than coding logic), accessible (browsers generate appropriate error messages for screen readers), and requires zero JavaScript. However, it has limitations: error messages are browser-controlled and cannot be easily customized, validation logic cannot handle complex cross-field dependencies (like "confirm password must match password"), and the visual presentation of error states varies across browsers.

**2. JavaScript Validation -- Full Control and Custom Logic:**

For validation needs that exceed HTML5's built-in capabilities, JavaScript provides the Constraint Validation API and complete programmatic control. The checkValidity() method on a form or individual input returns true if all constraints are satisfied and false otherwise. The reportValidity() method does the same but also triggers the browser's native error UI. The validity property on each input is an object with boolean properties indicating exactly what is wrong: valueMissing (required but empty), typeMismatch (wrong format for the input type), patternMismatch (does not match the pattern attribute), tooShort/tooLong (violates length constraints), rangeUnderflow/rangeOverflow (outside min/max range), and stepMismatch (does not match the step increment). The setCustomValidity() method lets you set your own error message that integrates with the browser's validation UI -- pass an empty string to clear the error. Beyond the Constraint Validation API, you can implement entirely custom validation logic: listening for the "input" or "change" event on each field, running your validation checks, and dynamically showing or hiding error messages. This approach gives you complete control over the timing, appearance, and content of error feedback, which is essential for polished user experiences.

**3. Validation Strategies -- When and How to Validate:**

There are several strategies for when validation occurs, and the best approach typically combines multiple strategies. Real-time (or "inline") validation checks each field as the user types or immediately after they leave the field (on the "blur" event), providing instant feedback. This is the most user-friendly approach because it catches errors immediately, but it can be annoying if validation triggers before the user has finished typing (for example, showing "invalid email" after the user has only typed the first character). A common refinement is to validate on blur (when the field loses focus) for the initial check, then switch to real-time validation (on each keystroke) once an error has been shown, so the user sees the error clear as they correct it. On-submit validation checks all fields when the user clicks the submit button, which is simpler to implement but provides a worse experience because the user must find and fix all errors after attempting to submit. Server-side validation is always required regardless of client-side validation, because any client-side check can be bypassed by a determined user (by disabling JavaScript, using browser developer tools, or sending requests directly with tools like curl). Client-side validation is for user experience; server-side validation is for data integrity and security.

**4. Accessible Validation -- Making Errors Clear to All Users:**

Validation is an area where accessibility is particularly important and frequently done poorly. Always associate error messages with their corresponding inputs using aria-describedby, so screen readers announce the error when the user focuses the field. Use aria-invalid="true" on inputs that have failed validation, so assistive technologies can communicate the error state. Provide clear, specific error messages that tell the user exactly what is wrong and how to fix it ("Password must be at least 8 characters" is far better than "Invalid input"). Never rely solely on color to indicate errors -- a red border is invisible to color-blind users, so always include a text message and/or an icon alongside the color change. Group related validation errors in a summary at the top of the form (with links to each invalid field) as well as inline next to each field, so users can quickly scan all issues. Use the aria-live="polite" attribute on error message containers so that screen readers announce dynamically added error messages without interrupting the user's current action.`,
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
					Content: `File uploads and the FormData API are essential capabilities for any web application that needs to handle user-submitted files -- profile photos, document attachments, image galleries, CSV imports, or any other binary content. While standard form submissions can handle text fields easily, files require special treatment because they are binary data that can be very large and need to be transmitted differently than simple text values. The FormData API provides a modern, programmatic way to construct form submissions in JavaScript, making it possible to upload files via AJAX (without a full page reload) and to dynamically build complex form payloads that mix text fields with file attachments.

**1. The File Input Element -- Giving Users a Way to Select Files:**

The file input (input type="file") is the HTML element that allows users to browse their file system and select files for upload. By default, it allows selecting a single file, but adding the multiple attribute lets users select several files at once (by holding Ctrl or Shift while clicking in the file picker dialog). The accept attribute constrains which file types the picker shows: accept="image/*" filters for all image types, accept=".pdf,.doc,.docx" filters for specific extensions, and accept="video/mp4,video/webm" filters for specific MIME types. While the accept attribute guides users toward valid files, it is purely a UI convenience -- it does not enforce validation, so you must always validate file types on both the client side and server side. The native file input looks different across browsers and is notoriously difficult to style with CSS, which is why many developers hide the actual input element (with display: none or the hidden attribute) and trigger it programmatically via a styled label or button, using the label's "for" attribute to link to the hidden input. When the user selects files, the input's files property returns a FileList object containing File objects, each with properties like name (the filename), size (in bytes), type (the MIME type), and lastModified (a timestamp).

**2. The FormData API -- Building Submissions Programmatically:**

The FormData object provides a way to construct key-value pairs representing form fields and their values, including files, which can then be sent via the Fetch API or XMLHttpRequest. You can create a FormData object from an existing form element (new FormData(formElement), which automatically captures all the form's current values) or build one from scratch. The append() method adds a new field: formData.append("username", "john") adds a text field, and formData.append("avatar", fileInput.files[0]) adds a file. You can append multiple values under the same key (useful for multi-file uploads under a single field name). The get() and getAll() methods retrieve values, set() replaces an existing key's value, delete() removes a key entirely, and has() checks for a key's existence. The entries() method returns an iterator for looping over all key-value pairs. When sending FormData with fetch(), do not set the Content-Type header manually -- the browser will automatically set it to "multipart/form-data" with the correct boundary string that separates the different parts of the payload. Setting the header manually will break the upload because the boundary will be missing.

**3. The FileReader API -- Processing Files on the Client Side:**

The FileReader API allows your JavaScript code to read the contents of files selected by the user, enabling client-side processing before (or instead of) uploading to a server. FileReader operates asynchronously through event-based callbacks. The readAsText() method reads a file as a text string -- useful for CSV files, JSON files, or any text-based content you want to parse or preview in the browser. The readAsDataURL() method reads a file as a Base64-encoded data URL -- this is the standard approach for showing image previews before upload, because you can set the resulting data URL as the src attribute of an img element. The readAsArrayBuffer() method reads a file as raw binary data, useful for processing binary formats or performing client-side analysis (like checking image dimensions or validating file signatures). URL.createObjectURL() provides a more performant alternative to readAsDataURL() for image previews -- it creates a temporary URL pointing directly to the file in memory without Base64 encoding, which is faster and uses less memory for large files.

**4. Best Practices for Robust File Upload Experiences:**

Always validate files on the client side before uploading -- check that the file type matches your requirements (by inspecting file.type or the file extension), the file size is within acceptable limits (comparing file.size against your maximum, like 5 * 1024 * 1024 for a 5MB limit), and the number of files does not exceed your maximum. Provide clear feedback about upload progress, which requires using XMLHttpRequest's upload.onprogress event (the Fetch API does not natively support upload progress tracking yet, though this is changing with newer browser APIs). Display a progress bar or percentage so users know the upload is working, especially for large files on slow connections. Handle errors gracefully -- network interruptions, server rejections, and timeout failures should all produce clear, user-friendly messages with an option to retry. Consider implementing drag-and-drop file upload using the HTML Drag and Drop API (listening for dragover and drop events on a designated drop zone), which provides a much better user experience than the default file input for applications where file upload is a primary action.`,
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
			ID:          1507,
			Title:       "Accessibility Basics",
			Description: "Make websites accessible to all users: semantic HTML, ARIA attributes, and keyboard navigation.",
			Order:       7,
			Lessons: []problems.Lesson{
				{
					Title: "Web Accessibility Fundamentals",
					Content: `Web accessibility (often abbreviated as "a11y" -- the "11" represents the eleven letters between the "a" and "y") is the practice of designing and developing websites so that people with disabilities can perceive, understand, navigate, and interact with them effectively. This is not a niche concern affecting a small minority -- the World Health Organization estimates that over one billion people worldwide (approximately 15% of the global population) live with some form of disability, and many more experience temporary or situational impairments (like a broken arm, bright sunlight making a screen hard to read, or a noisy environment making audio inaudible). Accessibility is simultaneously a legal requirement, a moral imperative, and a business advantage that improves the experience for all users.

**1. Why Accessibility Matters -- Legal, Ethical, and Business Imperatives:**

From a legal perspective, web accessibility is not optional in many jurisdictions. In the United States, the Americans with Disabilities Act (ADA) has been interpreted by courts to apply to websites, and organizations have faced significant lawsuits for inaccessible web experiences -- the number of ADA-related web accessibility lawsuits has grown dramatically, exceeding thousands per year. The European Union's EN 301 549 standard and the European Accessibility Act impose similar requirements across Europe. Section 508 of the Rehabilitation Act requires U.S. federal agencies to make their electronic content accessible. Beyond legal compliance, there is a profound ethical dimension: the web was designed to be universal, and excluding people with disabilities from accessing information, services, and opportunities that others take for granted is a form of discrimination. From a business perspective, accessible websites reach a larger audience (including the significant spending power of people with disabilities and the aging population), perform better in search engine rankings (because many accessibility practices overlap with SEO best practices like semantic HTML, proper heading structures, and descriptive alt text), and often provide a better experience for everyone -- curb cuts in sidewalks were designed for wheelchair users but benefit everyone from parents with strollers to delivery workers with carts, and the same "curb cut effect" applies to digital accessibility.

**2. Types of Disabilities and How They Affect Web Use:**

Understanding the diverse ways people interact with the web is essential for building accessible experiences. Visual impairments range from complete blindness (users navigate entirely with screen readers that read page content aloud and through keyboard commands) to low vision (users may zoom in significantly, use high-contrast modes, or use screen magnifiers) to color blindness (affecting approximately 8% of men and 0.5% of women, making color-only information invisible). Auditory impairments range from complete deafness to partial hearing loss, affecting the ability to perceive audio and video content without captions or transcripts. Motor impairments include conditions like paralysis, tremors, limited dexterity, or repetitive strain injuries that make using a mouse difficult or impossible -- these users rely on keyboard navigation, voice commands, switch devices, or eye-tracking technology. Cognitive impairments include learning disabilities (like dyslexia), attention deficit disorders, memory impairments, and intellectual disabilities -- these users benefit from clear language, consistent navigation, and distraction-free layouts. Many people experience multiple impairments simultaneously, and everyone experiences situational impairments (trying to use a phone one-handed while holding a coffee, or reading a screen in bright sunlight).

**3. WCAG Guidelines -- The International Standard for Web Accessibility:**

The Web Content Accessibility Guidelines (WCAG), published by the W3C's Web Accessibility Initiative (WAI), are the internationally recognized standard for web accessibility. WCAG is organized around four fundamental principles, remembered by the acronym POUR. Perceivable means information and user interface components must be presentable to users in ways they can perceive -- this includes providing text alternatives for non-text content (alt text for images), captions for audio/video, and ensuring content can be presented in different ways (like screen readers) without losing information. Operable means user interface components and navigation must be usable -- this includes keyboard accessibility for all functionality, providing enough time for users to read and interact, not designing content that causes seizures (no flashing elements more than three times per second), and providing clear navigation mechanisms. Understandable means information and operation of the user interface must be comprehensible -- this includes readable text, predictable behavior, and input assistance (like error identification and suggestions). Robust means content must be robust enough to be interpreted reliably by a wide variety of user agents, including assistive technologies. WCAG defines three conformance levels: Level A (minimum, addressing the most critical barriers), Level AA (the standard target for most organizations, addressing significant barriers), and Level AAA (the highest level, addressing more nuanced barriers). Most legal requirements and organizational targets aim for Level AA compliance.

**4. Key Accessibility Practices -- Practical Steps Every Developer Should Take:**

Semantic HTML is the foundation of accessibility -- using the correct HTML elements (nav for navigation, button for buttons, heading elements for headings) gives assistive technologies the information they need to convey page structure and functionality to users. A proper heading hierarchy (h1 through h6 without skipping levels) allows screen reader users to navigate by headings, which is one of their primary navigation methods. Every informational image must have descriptive alt text that conveys the image's meaning or purpose (not just "image" or "photo"), while purely decorative images should have an empty alt attribute (alt="") so screen readers skip them entirely. All interactive functionality must be operable via keyboard alone -- this means every button, link, form field, menu, and custom widget must be focusable, activatable, and navigable without a mouse. Color contrast between text and its background must meet WCAG minimums (4.5:1 ratio for normal text, 3:1 for large text) to ensure readability for users with low vision or color blindness. ARIA (Accessible Rich Internet Applications) attributes supplement semantic HTML when native elements do not fully convey an element's role or state, but the first rule of ARIA is "do not use ARIA if you can use a native HTML element instead" -- misused ARIA can make accessibility worse, not better.`,
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
					Content: `ARIA (Accessible Rich Internet Applications) is a set of HTML attributes defined by the W3C that provide additional semantic information to assistive technologies when native HTML elements alone are insufficient to convey the purpose, state, or behavior of interactive components. Keyboard navigation, closely related to ARIA, is the practice of ensuring that every interactive element on your page can be reached, activated, and operated using only the keyboard -- no mouse required. Together, ARIA and keyboard navigation are the two pillars that make complex, JavaScript-driven web applications accessible to users who rely on screen readers, keyboard-only navigation, switch devices, and other assistive technologies.

**1. ARIA Attributes -- Bridging the Gap Between Custom Widgets and Assistive Technologies:**

Modern web applications are full of custom interactive components that go far beyond basic HTML: dropdown menus, modal dialogs, tab panels, accordions, autocomplete fields, drag-and-drop interfaces, and more. These components are typically built from generic elements like div and span styled to look interactive, but without additional information, screen readers have no way to know that a styled div is acting as a button, a tab, or a menu item. ARIA attributes provide that missing information. The aria-label attribute assigns an accessible name to an element when no visible text label exists -- for example, a close button that only displays an "X" icon needs aria-label="Close dialog" so screen readers announce its purpose. The aria-labelledby attribute points to another element whose text content serves as the label, useful when the label is visible on the page but not programmatically associated with the control. The aria-describedby attribute links to an element providing supplementary description (like a password hint or field instructions), which the screen reader announces after the label. The aria-hidden="true" attribute hides elements from the accessibility tree entirely, useful for decorative icons, duplicate content, or visually hidden elements that would confuse screen reader users. The aria-live attribute designates regions whose content may change dynamically (like notification banners, chat messages, or status updates), telling screen readers to announce changes as they happen -- "polite" waits for the user to finish their current action before announcing, while "assertive" interrupts immediately. The role attribute overrides or clarifies an element's semantic role (role="button", role="dialog", role="tablist", role="alert", etc.), telling assistive technologies how to present and interact with the element. However, the cardinal rule of ARIA is: do not use ARIA if a native HTML element or attribute will do the job. A native button element is always preferable to a div with role="button", because the native element comes with built-in keyboard handling, focus management, and form submission behavior that you would otherwise have to reimplement manually.

**2. Keyboard Navigation -- Ensuring Universal Operability:**

Keyboard navigation follows well-established conventions that users of assistive technologies rely on. The Tab key moves focus forward through interactive elements (links, buttons, form inputs, and elements with tabindex), while Shift+Tab moves focus backward. Enter activates the focused element (follows links, submits buttons), and Space also activates buttons and toggles checkboxes. Arrow keys navigate within composite widgets like dropdown menus, radio button groups, tab lists, and tree views. The Escape key dismisses overlays like modals, dropdowns, and tooltips. These conventions are not just guidelines -- they are deeply ingrained expectations that keyboard users depend on. When you build custom widgets, you must implement these keyboard interactions manually using JavaScript event listeners for the "keydown" event, checking event.key for values like "Enter", "Space" (actually " "), "Escape", "ArrowDown", "ArrowUp", "ArrowLeft", and "ArrowRight". Failing to implement expected keyboard behaviors makes your custom components effectively unusable for keyboard-only users, even if they look and work perfectly with a mouse.

**3. Focus Management -- Controlling the User's Attention:**

Focus management is the practice of programmatically controlling which element has keyboard focus, and it is critical for maintaining a coherent navigation experience in dynamic applications. The tabindex attribute controls whether and in what order elements appear in the tab sequence: tabindex="0" makes any element focusable and includes it in the natural tab order (essential for custom interactive widgets built from non-interactive elements), tabindex="-1" makes an element focusable via JavaScript (element.focus()) but removes it from the tab order (useful for elements that should receive focus programmatically but not via Tab), and positive tabindex values (tabindex="1", tabindex="2") create a custom tab order -- but this is almost always a bad practice because it creates a confusing experience that diverges from the visual layout. Visible focus indicators (the outline that appears around focused elements) are critically important and should never be removed with outline: none without providing an alternative visible focus style -- users who navigate by keyboard have no way to know where they are on the page without a focus indicator. Focus trapping is essential for modal dialogs: when a modal opens, focus should move to the first focusable element inside the modal, Tab and Shift+Tab should cycle only through the modal's focusable elements (never escaping to the page behind), and when the modal closes, focus should return to the element that triggered the modal. Without focus trapping, keyboard users can Tab behind the modal into invisible page content, getting completely lost.

**4. Best Practices for Accessible Interactive Components:**

Every interactive element must be keyboard accessible -- if it responds to a mouse click, it must also respond to Enter and/or Space key presses. Ensure a logical tab order that follows the visual layout of the page, typically left-to-right and top-to-bottom in left-to-right languages. Implement skip links at the top of the page (hidden until focused) that let keyboard users jump directly to the main content, bypassing repetitive navigation menus -- this is one of the most impactful single accessibility improvements you can make. Never create keyboard traps -- situations where a user can Tab into an element or widget but cannot Tab out of it. Test your pages with a screen reader (VoiceOver on macOS, NVDA on Windows, or TalkBack on Android) at least once during development -- the experience of hearing your page read aloud often reveals issues that are invisible in visual testing. Use automated accessibility testing tools like axe, Lighthouse, or WAVE as part of your development workflow to catch common issues, but remember that automated tools only catch about 30-40% of accessibility issues -- manual testing with keyboard navigation and screen readers is irreplaceable.`,
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
			ID:          1508,
			Title:       "Browser APIs",
			Description: "Explore browser APIs: Local Storage, Fetch API, Geolocation, and more.",
			Order:       8,
			Lessons: []problems.Lesson{
				{
					Title: "Storage APIs",
					Content: `Browser storage APIs allow web applications to persist data directly on the user's device, enabling experiences that remember user preferences, cache content for offline use, maintain shopping cart state across sessions, and reduce server round-trips by storing frequently accessed data locally. Before these APIs existed, the only client-side storage option was cookies -- tiny text strings originally designed for server-side session tracking, not for general-purpose data storage. Modern browsers provide several storage mechanisms, each designed for different use cases, capacity needs, and lifetime requirements. Understanding when to use each one is a critical skill for building performant, user-friendly web applications.

**1. localStorage -- Persistent Key-Value Storage:**

localStorage provides a simple, synchronous key-value store that persists data across browser sessions -- meaning the data survives browser restarts, computer reboots, and even operating system updates. It is scoped to the origin (protocol + domain + port), so data stored on https://example.com is completely isolated from data on https://other.com or even http://example.com (different protocol). The API is beautifully straightforward: setItem(key, value) stores a string, getItem(key) retrieves it (or returns null if the key does not exist), removeItem(key) deletes a specific key, and clear() removes all stored data for the origin. The critical limitation is that both keys and values must be strings -- to store objects or arrays, you must serialize them with JSON.stringify() before storing and deserialize with JSON.parse() when retrieving. The storage capacity is typically 5-10MB per origin (varies by browser), which is more than enough for user preferences, UI state, and moderate amounts of cached data, but insufficient for large datasets. Because the API is synchronous, reading and writing localStorage blocks the main thread, which can cause performance issues if you are storing or retrieving large amounts of data -- this is rarely a problem in practice, but it is worth knowing. Common use cases include storing user theme preferences (dark/light mode), remembering the last-viewed page or scroll position, caching API responses to reduce network requests, and storing authentication tokens (though this has security implications -- see below).

**2. sessionStorage -- Temporary, Tab-Scoped Storage:**

sessionStorage shares the exact same API as localStorage (setItem, getItem, removeItem, clear) but with a crucial difference in lifetime and scope: data stored in sessionStorage is automatically cleared when the browser tab or window is closed, and each tab has its own independent sessionStorage even when multiple tabs are open on the same origin. This makes sessionStorage ideal for data that should not persist beyond the current browsing session: multi-step form wizards (preserving form data as the user navigates between steps without committing to permanent storage), temporary UI state that should reset when the user starts a new session, sensitive information that should not linger on the device, and one-time tutorial or onboarding flows. The tab isolation is particularly important: if a user has two tabs open on the same site, each tab's sessionStorage is completely independent, preventing one tab's state from interfering with another's. Like localStorage, sessionStorage has a capacity of approximately 5-10MB per origin per tab.

**3. IndexedDB -- The Heavy-Duty Client-Side Database:**

For storage needs that exceed localStorage's capabilities -- larger datasets, complex queries, or structured data -- IndexedDB provides a full-featured, asynchronous, transactional object database built into the browser. Unlike localStorage's simple key-value strings, IndexedDB stores JavaScript objects directly (no serialization needed), supports indexes for efficient querying, handles large amounts of data (hundreds of megabytes or more, with user permission), and operates asynchronously so it does not block the main thread. The trade-off is API complexity: IndexedDB uses a request-based, event-driven API that is significantly more verbose than localStorage. You must open a database, create object stores (similar to tables), define indexes, and wrap operations in transactions. For this reason, many developers use wrapper libraries like idb (by Jake Archibald) or Dexie.js that provide a cleaner Promise-based or async/await-compatible API. IndexedDB is the backbone of offline-capable web applications (Progressive Web Apps), powering features like offline email clients, local caching of large datasets, and client-side search functionality.

**4. Cookies -- The Original (and Still Relevant) Storage Mechanism:**

Cookies are the oldest client-side storage mechanism, predating the Web Storage API by over a decade. They are small text strings (limited to approximately 4KB) that are automatically sent to the server with every HTTP request to the matching domain and path -- this automatic transmission is what makes them essential for authentication (session tokens, JWT tokens) and server-side user tracking, but it is also a performance concern because every cookie increases the size of every HTTP request. Cookies support expiration dates (expires or max-age attributes), domain and path scoping, the secure flag (only transmitted over HTTPS), the httpOnly flag (inaccessible to JavaScript, protecting against XSS attacks), and the sameSite attribute (controls whether cookies are sent with cross-site requests, protecting against CSRF attacks). While localStorage and sessionStorage have largely replaced cookies for client-side data storage, cookies remain the standard mechanism for authentication tokens and server-side session management because of their automatic transmission with requests and their httpOnly security capability. When choosing a storage mechanism, consider: use cookies for authentication and data the server needs, localStorage for persistent client-side preferences and cache, sessionStorage for temporary per-tab data, and IndexedDB for large or complex data that needs to be queried.`,
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
					Content: `The Fetch API is the modern, Promise-based interface for making HTTP requests from the browser, and it is one of the most important APIs you will use as a frontend developer. Nearly every non-trivial web application needs to communicate with a server -- loading data to display, submitting user input, authenticating users, uploading files, or synchronizing state between the client and a backend. Before the Fetch API, developers relied on XMLHttpRequest (XHR), a callback-based API with a verbose, awkward syntax that made complex request flows difficult to read and maintain. Fetch replaced XHR with a cleaner, Promise-based design that integrates beautifully with async/await syntax, making asynchronous HTTP operations almost as readable as synchronous code.

**1. How Fetch Works -- The Promise-Based Request Model:**

At its simplest, fetch() takes a URL and returns a Promise that resolves to a Response object. The call itself does not give you the data directly -- the Response object represents the entire HTTP response, including status code, headers, and a body that must be explicitly read. This two-step process (initiate request, then read body) is intentional: it allows you to inspect the response status and headers before committing to reading the potentially large response body. To extract JSON data, you call response.json() (which returns another Promise). For text content, use response.text(). For binary data (like images), use response.blob() or response.arrayBuffer(). A critical gotcha that trips up many beginners: fetch() only rejects its Promise on network failures (like no internet connection or DNS errors) -- it does not reject on HTTP error status codes like 404 (Not Found) or 500 (Internal Server Error). You must manually check response.ok (which is true for status codes 200-299) or inspect response.status to detect and handle HTTP errors. This design choice reflects the fact that an HTTP 404 is still a successful HTTP transaction -- the server responded -- even though the application may consider it an error.

**2. HTTP Methods -- The Vocabulary of Client-Server Communication:**

HTTP methods (also called "verbs") tell the server what operation you want to perform on a resource. GET retrieves data without modifying anything on the server -- it is the default method for fetch() and should be used for all read-only operations (loading user profiles, fetching search results, retrieving product lists). GET requests should be idempotent (making the same request multiple times produces the same result) and should never cause side effects on the server. POST sends data to the server to create a new resource or trigger a server-side action -- used for submitting forms, creating new records, or initiating processes. PUT replaces an entire resource with the provided data -- used for full updates where you send the complete new state of the resource. PATCH performs a partial update, sending only the fields that changed -- more efficient than PUT when you only need to modify one or two properties of a large resource. DELETE removes a resource from the server. Understanding which method to use for each operation is fundamental to building applications that interact correctly with RESTful APIs, and it affects caching behavior, security, and idempotency.

**3. Request Configuration -- Customizing Every Aspect of the Request:**

The second argument to fetch() is an options object that lets you configure every aspect of the request. The method property specifies the HTTP method (defaults to "GET"). The headers property is an object (or Headers instance) containing HTTP headers -- most commonly "Content-Type": "application/json" for JSON request bodies, and "Authorization": "Bearer <token>" for authenticated requests. The body property contains the request payload: for JSON APIs, you pass JSON.stringify(data); for form submissions, you pass a FormData object; for file uploads, you also use FormData (and importantly, do not set the Content-Type header manually, as the browser needs to set it with the correct multipart boundary). The mode property controls CORS (Cross-Origin Resource Sharing) behavior: "cors" (the default) allows cross-origin requests with proper server headers, "same-origin" restricts to same-origin requests, and "no-cors" makes opaque requests (limited use). The credentials property controls whether cookies and HTTP authentication headers are included: "same-origin" (the default) sends credentials only to the same origin, "include" sends credentials even for cross-origin requests (required for authenticated cross-origin API calls), and "omit" never sends credentials.

**4. Error Handling and Advanced Patterns -- Building Robust Data Fetching:**

Robust error handling is essential for a good user experience. Wrap fetch calls in try/catch blocks (when using async/await) to catch network errors. Always check response.ok before parsing the body to catch HTTP errors. Parse error response bodies (many APIs return error details in JSON format) to provide specific, helpful error messages to users. Implement timeout handling, since fetch() has no built-in timeout -- the common pattern is to race the fetch Promise against a setTimeout Promise using Promise.race(), rejecting if the timer fires first. For retry logic, implement exponential backoff (wait 1 second, then 2, then 4) for transient failures like network glitches or server overloads, but never retry requests that are not idempotent (you do not want to accidentally create duplicate records by retrying a POST). The AbortController API integrates with fetch() to cancel in-flight requests -- create a controller, pass its signal to the fetch options, and call controller.abort() to cancel. This is essential for search-as-you-type features (cancel the previous request when the user types a new character) and for cleaning up requests when a component unmounts in frameworks like React.`,
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
					Content: `The Geolocation, Notification, and Clipboard APIs are powerful browser capabilities that extend web applications beyond simple content display, enabling location-aware features, proactive user communication, and seamless clipboard integration. These APIs share an important characteristic: they all require explicit user permission before they can be used, reflecting the browser's role as a privacy guardian between web applications and the user's device. Understanding how to request permissions gracefully, handle denial without breaking your application, and use these capabilities responsibly is just as important as understanding their technical interfaces.

**1. The Geolocation API -- Building Location-Aware Experiences:**

The Geolocation API allows your web application to request the user's geographic coordinates, enabling features like showing nearby stores on a map, providing location-specific weather forecasts, auto-filling city and region in address forms, or calculating distances and directions. When you call navigator.geolocation.getCurrentPosition(), the browser prompts the user with a permission dialog asking whether they want to share their location with your site. If they grant permission, your success callback receives a Position object containing coordinates (latitude, longitude, accuracy in meters, and optionally altitude, heading, and speed). If they deny permission or if geolocation is unavailable (like on a desktop without GPS), your error callback receives a PositionError with a code indicating what went wrong: PERMISSION_DENIED (user said no), POSITION_UNAVAILABLE (device cannot determine location), or TIMEOUT (took too long). The watchPosition() method is similar but continuously monitors the user's location, calling your callback each time it changes -- ideal for navigation or fitness tracking applications. The options object lets you configure enableHighAccuracy (true uses GPS if available, which is more precise but slower and more battery-draining), timeout (maximum time to wait for a position), and maximumAge (how old a cached position can be before a fresh one is required). Always check for API availability first with "geolocation" in navigator, and always provide a meaningful fallback experience for users who deny permission or whose devices lack geolocation capabilities -- for example, letting them manually enter a location.

**2. The Notification API -- Engaging Users Beyond the Browser Tab:**

The Notification API lets your web application display native operating system notifications even when the user is not actively looking at your page -- think of new message alerts, order status updates, breaking news, or reminder notifications. This is a powerful engagement tool, but it must be used responsibly because notification fatigue causes users to block all notifications from your site. The API follows a two-step process: first, request permission with Notification.requestPermission() (which returns a Promise resolving to "granted", "denied", or "default"), and then create notifications with new Notification(title, options). The options object supports a body (the notification's main text), icon (a small image displayed alongside the text), badge (a smaller monochrome icon for mobile), tag (a string identifier that prevents duplicate notifications -- a new notification with the same tag replaces the previous one), requireInteraction (keeps the notification visible until the user explicitly dismisses it), and actions (buttons within the notification for quick responses). Notification events (onclick, onclose, onerror) let you respond when the user interacts with the notification -- typically, clicking a notification should bring the user to the relevant content in your application. For persistent notifications that survive page navigations and work even when the page is closed, you need to use Service Workers with the showNotification() method on the ServiceWorkerRegistration, which is the foundation of push notification functionality in Progressive Web Apps.

**3. The Clipboard API -- Seamless Copy and Paste Integration:**

The Clipboard API provides a modern, Promise-based interface for reading from and writing to the user's clipboard, replacing the older, less reliable document.execCommand("copy") approach. The navigator.clipboard.writeText(text) method copies a string to the clipboard -- perfect for "copy to clipboard" buttons next to code snippets, API keys, referral links, or sharing URLs. The navigator.clipboard.readText() method reads text from the clipboard -- useful for "paste from clipboard" functionality in applications that need to process pasted content. For richer content, the write() and read() methods support ClipboardItem objects that can contain multiple representations (plain text, HTML, images) of the same content. The Clipboard API requires a secure context (HTTPS) and user interaction (the call must originate from a user gesture like a click) for security reasons -- you cannot silently read or write the clipboard in the background. Reading from the clipboard additionally requires explicit permission, since clipboard contents may contain sensitive information. When implementing copy functionality, always provide visual feedback to the user (like changing the button text to "Copied!" for a few seconds) so they know the operation succeeded.

**4. Best Practices for Permission-Gated APIs:**

Never request permissions on page load -- this is the single most common mistake developers make with these APIs, and it almost guarantees the user will deny the request. Instead, request permissions in context, when the user takes an action that clearly motivates the need: ask for location when the user clicks "Find stores near me," ask for notification permission when they enable a "notify me about replies" feature, and trigger clipboard access when they click a "copy" button. Always handle permission denial gracefully -- your application should work without the permission, just with reduced functionality. Provide clear explanations of why you need the permission and what benefit the user will receive (for example, "Enable notifications to get real-time updates when someone responds to your comment"). Cache permission states to avoid repeated prompts, and remember that once a user denies a permission in modern browsers, you cannot programmatically prompt again -- the user must manually change the permission in browser settings, so make it easy for them to find this option. Respect user privacy above all: only request the minimum data you need, do not store sensitive data (like precise location) longer than necessary, and be transparent about how you use the information.`,
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
			ID:          1509,
			Title:       "ES6+ Features",
			Description: "Modern JavaScript features: let/const, arrow functions, destructuring, modules, and more.",
			Order:       9,
			Lessons: []problems.Lesson{
				{
					Title: "ES6 Core Features",
					Content: `ES6, officially known as ECMAScript 2015, was the single most transformative update in JavaScript's history -- a quantum leap that modernized the language and addressed long-standing pain points that had frustrated developers for years. Before ES6, JavaScript had not received a major update since ES5 in 2009, and many of its design patterns felt dated compared to other modern languages. ES6 introduced features that fundamentally changed how JavaScript code is written, making it more readable, less error-prone, and more expressive. If you have ever seen modern JavaScript code that looks cleaner and more elegant than the older "var-heavy" code in legacy tutorials, ES6 is the reason why. Understanding these core features is non-negotiable for any modern JavaScript developer, as they are used universally in contemporary codebases, frameworks, and libraries.

**1. let and const -- The End of var's Reign:**

The introduction of let and const was arguably the most impactful change in ES6, because it fixed one of JavaScript's most confusing behaviors: variable scoping with var. Variables declared with var are function-scoped, meaning they are accessible throughout the entire function they are defined in, regardless of block boundaries like if statements, loops, or try-catch blocks. This leads to subtle bugs where a variable "leaks" out of a loop or conditional block and is unexpectedly accessible (or worse, modified) elsewhere. The let keyword introduced true block scoping: a variable declared with let inside an if block or for loop only exists within those curly braces. The const keyword adds the additional constraint that the variable binding cannot be reassigned after initialization -- attempting to do so throws an error, catching accidental reassignments at development time. A crucial nuance: const prevents reassignment of the variable binding, not mutation of the value. If you assign an object to a const, you can still modify the object's properties (add, change, or delete them); you just cannot point the variable at a different object entirely. Both let and const also respect the "temporal dead zone" -- accessing a let or const variable before its declaration line throws a ReferenceError, unlike var which would silently return undefined. The modern best practice is simple and universal: use const by default for every variable, and only switch to let when you have a specific, identifiable reason to reassign (like a loop counter or a variable updated in a conditional). Never use var in new code.

**2. Arrow Functions -- Concise Syntax with Predictable this Binding:**

Arrow functions provide a shorter syntax for writing function expressions and solve one of JavaScript's most notorious gotchas: the unpredictable behavior of the "this" keyword. In traditional functions, "this" is determined by how the function is called, not where it is defined -- a common source of bugs in event handlers, callbacks, and methods passed as arguments. Arrow functions do not have their own "this" binding; instead, they inherit "this" from the enclosing lexical scope (the scope where the arrow function is defined), making their behavior predictable and eliminating the need for hacks like var self = this or .bind(this). The syntax is also more concise: for a single expression, you can omit the curly braces and the return keyword (the expression is implicitly returned), turning a three-line function into a single, readable expression. For example, const double = (x) => x * 2 is equivalent to the more verbose const double = function(x) { return x * 2; }. When the arrow function has exactly one parameter, you can even omit the parentheses around it: const greet = name => "Hello, " + name. Arrow functions cannot be used as constructors (calling new on an arrow function throws an error) and do not have their own arguments object, which are features of traditional functions that are occasionally needed. The rule of thumb: use arrow functions for callbacks, array methods, and short utility functions; use traditional function declarations for methods that need their own "this" (like object methods or class methods) and for top-level named functions that benefit from hoisting.

**3. Template Literals -- Readable String Construction:**

Template literals, enclosed in backtick characters instead of single or double quotes, solve the age-old pain of string concatenation in JavaScript. Before template literals, building a string that included variable values required awkward concatenation with the + operator: "Hello, " + name + "! You have " + count + " messages." With template literals, you embed expressions directly using ${} syntax: ` + "`" + `Hello, ${name}! You have ${count} messages.` + "`" + `. This is not just shorter -- it is dramatically more readable, especially for complex strings with multiple embedded values or expressions. Template literals also support multi-line strings natively: you can press Enter inside a template literal and the line break is preserved in the output, eliminating the need for \n escape sequences or string concatenation across lines. Inside ${}, you can place any JavaScript expression -- not just variables, but function calls, arithmetic, ternary operators, and even other template literals. Tagged templates are an advanced feature where you prefix a template literal with a function name (like html` + "`" + `<p>${content}</p>` + "`" + `), and the function receives the string parts and expression values separately, enabling powerful use cases like CSS-in-JS libraries (styled-components), internationalization, and safe HTML escaping.

**4. Destructuring and Default Parameters -- Elegant Data Extraction:**

Destructuring assignment is a syntax that lets you unpack values from arrays and properties from objects into distinct variables in a single, readable statement. Object destructuring uses property name matching: const { name, age, city } = person extracts three properties from the person object into three separate variables. You can rename variables during extraction (const { name: fullName } = person), provide default values for missing properties (const { role = "user" } = person), and destructure nested objects in a single statement. Array destructuring uses position-based assignment: const [first, second, ...rest] = myArray extracts the first two elements into named variables and collects the remainder into an array called rest. You can skip elements with empty slots (const [, , third] = myArray). Destructuring is used everywhere in modern JavaScript: in function parameters (function greet({ name, age }) { ... } makes the expected input self-documenting), in import statements, in React component props, and in loop iterations over arrays of objects. Default parameters complement destructuring by letting you specify fallback values directly in the function signature: function createUser(name, role = "viewer", active = true) means callers only need to provide the name -- role and active will use their defaults if not specified. This eliminates the old pattern of manually checking for undefined inside the function body.`,
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
					Content: `Beyond the core ES6 features covered in the previous lesson, modern JavaScript has continued to evolve with powerful additions that address common programming challenges like immutable data operations, asynchronous programming, code organization, object-oriented design, and specialized data structures. These features are not merely syntactic sugar -- they represent fundamentally better tools for solving real-world problems, and they are universally used in modern codebases, frameworks, and libraries. Mastering them is essential for writing clean, maintainable, professional JavaScript code.

**1. Spread and Rest Operators -- The Versatile Three Dots:**

The spread operator (...) and rest operator (...) use the same syntax (three dots) but serve opposite purposes depending on context. The spread operator expands an iterable (like an array or object) into its individual elements. For arrays, [...arr1, ...arr2] creates a new array combining both arrays without modifying either original -- this immutable approach is crucial in React and other frameworks where mutating state directly causes bugs. For objects, { ...obj1, ...obj2 } creates a new object merging both, with properties from obj2 overriding matching properties from obj1 -- this is the standard pattern for creating modified copies of objects (like { ...user, name: "New Name" } to update just the name while preserving all other properties). Spread also lets you pass array elements as individual function arguments: Math.max(...numbers) finds the maximum of an array. The rest operator does the reverse: it collects multiple individual elements into a single array. In function parameters, function log(first, ...remaining) captures the first argument separately and bundles all remaining arguments into an array. In destructuring, const [head, ...tail] = myArray extracts the first element and collects the rest. A critical distinction: spread creates shallow copies, not deep copies. If your object contains nested objects, the nested references are shared, not duplicated. For true deep cloning, use structuredClone() in modern environments.

**2. Promises and Async/Await -- Taming Asynchronous Complexity:**

Asynchronous operations -- network requests, file reads, timers, database queries -- are fundamental to JavaScript programming, and Promises provide a structured way to handle their eventual completion or failure. A Promise represents a value that may not be available yet but will be resolved (successfully) or rejected (with an error) in the future. You chain .then() to handle success and .catch() to handle errors, and each .then() returns a new Promise, enabling clean chains of sequential async operations. However, deeply nested .then() chains can still become hard to read (sometimes called "promise spaghetti"), which is where async/await syntax provides a transformative improvement. An async function automatically returns a Promise, and the await keyword pauses execution within that function until a Promise resolves, making asynchronous code read almost like synchronous code. Error handling uses familiar try/catch blocks instead of .catch() chains. Promise.all() takes an array of Promises and resolves when all complete (or rejects if any fail), perfect for running multiple independent requests in parallel. Promise.race() resolves as soon as the first Promise in the array settles (either resolves or rejects), useful for implementing timeouts. Promise.allSettled() waits for all Promises to complete regardless of success or failure, returning the status and value/reason for each -- ideal when you need results from multiple independent operations that might partially fail.

**3. Classes -- Object-Oriented JavaScript Made Clear:**

The class syntax introduced in ES6 provides a cleaner, more familiar way to create objects and implement inheritance in JavaScript, replacing the confusing prototype-based patterns that previously required deep understanding of JavaScript's internal mechanics. A class definition includes a constructor method (called when you create a new instance with the "new" keyword) that initializes instance properties, regular methods that all instances share, static methods (prefixed with the "static" keyword) that belong to the class itself rather than instances and are typically used for utility functions or factory methods, and getter/setter methods for computed properties. Inheritance uses the "extends" keyword: class Student extends Person means Student inherits all of Person's methods and properties. The super keyword calls the parent class's constructor or methods. Private class fields (prefixed with #, like #password) are a newer addition that provides true encapsulation -- private fields cannot be accessed from outside the class, unlike the older convention of prefixing "private" properties with an underscore (which was merely a naming convention with no enforcement). It is important to understand that classes are syntactic sugar over JavaScript's prototype-based inheritance -- they do not introduce a new object model. The prototype chain still underpins class behavior, and understanding this is important for debugging and advanced patterns.

**4. Map and Set -- Purpose-Built Collection Data Structures:**

While plain objects and arrays handle most data storage needs, Map and Set provide specialized collection types that solve specific problems more elegantly and efficiently. A Map is a key-value collection where keys can be any type -- objects, functions, numbers, even NaN -- unlike plain objects where keys are always coerced to strings. Maps maintain insertion order during iteration, provide a convenient .size property (unlike objects which require Object.keys(obj).length), and perform better for frequent additions and deletions. A Set is a collection of unique values -- any duplicate added is silently ignored. This makes Set the simplest way to remove duplicates from an array: [...new Set(myArray)]. Sets are also more efficient than arrays for checking membership: set.has(value) is O(1) constant time, while array.includes(value) is O(n) linear time. WeakMap and WeakSet are specialized variants where keys (WeakMap) or values (WeakSet) are held as "weak references," meaning they do not prevent garbage collection -- if the key object has no other references, it can be garbage collected even though it is in the WeakMap. This makes them ideal for caching computed results associated with DOM elements or objects without causing memory leaks.`,
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
					Content: `ES6 modules and iterators represent two features that transformed JavaScript from a language limited to monolithic script files into a modern, modular programming language capable of powering large-scale applications. Modules provide the ability to organize code into self-contained, reusable files with explicit dependencies, while iterators and generators provide a standardized protocol for stepping through sequences of values. Together, they enable the kind of clean, maintainable code architecture that professional JavaScript applications demand.

**1. ES6 Modules -- Organizing Code Into Manageable, Reusable Units:**

Before ES6 modules, JavaScript had no native way to split code across multiple files with proper dependency management. Developers relied on workarounds like immediately-invoked function expressions (IIFEs), the module pattern, or third-party module systems like CommonJS (used in Node.js with require()) and AMD (used with RequireJS for browser code). ES6 introduced a native module system with import and export keywords that the JavaScript engine understands directly, enabling static analysis (tools can determine dependencies at build time without running the code), tree shaking (bundlers can eliminate unused exports to reduce file size), and clear, explicit dependency graphs. Each module is its own file with its own scope -- variables, functions, and classes defined in a module are private by default and only accessible to other files if explicitly exported. This encapsulation prevents naming conflicts, reduces global scope pollution, and makes it clear which parts of a module are its "public API."

**2. Export and Import Syntax -- The Vocabulary of Module Communication:**

There are two types of exports: named exports and default exports. Named exports allow you to export multiple values from a single module, each identified by its name: export const add = (a, b) => a + b exports the add function by name, and importing files must use the exact same name inside curly braces: import { add } from "./math.js". You can rename during import with the "as" keyword: import { add as sum } from "./math.js". A module can have as many named exports as needed, making them ideal for utility libraries that provide multiple related functions. A default export specifies a single "main" value for the module: export default function multiply(a, b) { return a * b; }. Importing a default export does not use curly braces, and you can assign any name you want: import multiply from "./math.js" or import myMultiplier from "./math.js". A module can have at most one default export (alongside any number of named exports). You can import everything from a module as a namespace object: import * as math from "./math.js" gives you math.add, math.subtract, etc. Dynamic imports using the import() function (note the parentheses -- this is a function call, not the import declaration) return a Promise that resolves to the module, enabling lazy loading: const module = await import("./heavy-feature.js"). This is powerful for code splitting in large applications, loading features only when the user needs them.

**3. Iterators -- A Standard Protocol for Sequential Access:**

The iterator protocol defines a standard way to produce a sequence of values, one at a time. Any object that implements a next() method returning objects of the form { value: someValue, done: false } (or { done: true } when the sequence is exhausted) is an iterator. Many built-in JavaScript types are "iterable" -- they implement the Symbol.iterator method that returns an iterator. Arrays, strings, Maps, Sets, NodeLists, and arguments objects are all iterable, which is why they work with for...of loops, spread syntax, and destructuring. The for...of loop is the clean, modern way to iterate over any iterable: for (const item of myArray) reads each value in sequence without needing index management. You can make any custom object iterable by implementing Symbol.iterator, which lets it work with for...of, spread, and destructuring -- for example, making a Range object that yields numbers from a start value to an end value. Understanding the iterator protocol is important because it underpins many modern JavaScript features and libraries: async iterators (for await...of) extend the protocol for asynchronous data sources like streams, database cursors, or paginated API results.

**4. Generators -- Functions That Pause and Resume:**

Generator functions, declared with the function* syntax (note the asterisk), are a special type of function that can pause execution midway through and resume later, yielding multiple values over time rather than computing and returning a single result. When you call a generator function, it does not execute the function body immediately -- instead, it returns a generator object (which is both an iterator and an iterable). Each call to the generator's next() method runs the function body until it hits a yield expression, which pauses execution and returns the yielded value. The next call to next() resumes from where it paused. This makes generators ideal for producing sequences of values lazily (on demand) without computing them all upfront: a Fibonacci generator can produce an infinite sequence without running out of memory, because each value is computed only when requested. Generators can also receive values: next(someValue) sends a value back into the generator, which becomes the result of the yield expression that paused it -- this enables powerful two-way communication patterns. Generators pair naturally with for...of loops: for (const value of myGenerator()) iterates through all yielded values until the generator returns. In practice, generators are used for lazy evaluation of large or infinite sequences, implementing custom iterables, controlling asynchronous flow (though async/await has largely replaced this use case), and building cooperative concurrency patterns. While generators are less commonly used directly in application code than other ES6 features, understanding them deepens your comprehension of JavaScript's iteration model and is essential for advanced library and framework development.`,
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
