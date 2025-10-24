# Code Style and Conventions

## General Principles
- **Educational Clarity First**: Code should be clear and educational for M.Tech students
- **Progressive Learning**: Start with basic concepts, build complexity gradually
- **Hands-On Focus**: Every theoretical concept needs practical implementation
- **Beginner-Friendly**: Assume diverse programming backgrounds

## Python Code Style
- Use descriptive variable names for educational clarity
- Include explanations suitable for M.Tech students
- Follow standard Python PEP 8 conventions
- Type hints are optional but encouraged for clarity

## Neural Network Patterns
```python
# Standard model creation pattern
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(rate),
    tf.keras.layers.Dense(output_units, activation='softmax')
])

# Standard compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Image Processing Patterns
- Convert to grayscale first
- Normalize to [0,1] range
- Standard resize to 28x28 for MNIST
- Use bounding box detection for preprocessing

## File Organization
- Tutorial tasks numbered T1-T15
- Weekly plans in course_planning/weekly_plans/
- Lab exercises in labs/
- Documentation in markdown format

## Documentation Standards
- Use markdown for all documentation
- Include learning objectives in each tutorial
- Provide both theory and practice sections
- Add troubleshooting sections for common issues

## Important Notes
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files
- NEVER proactively create documentation files unless requested
- Do what has been asked; nothing more, nothing less