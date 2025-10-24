# Special Guidelines and Important Notes

## Critical Requirements

### Chat History System (MANDATORY)
- **ALWAYS** ask user for current date/time before creating timestamped files
- **NEVER** use system date/time
- Read docs/chat_history/CLAUDE.md at session start
- Document all significant Q&A exchanges
- Use morning startup and end-of-day ceremony templates

### File Management Rules
- **NEVER** create files unless absolutely necessary
- **ALWAYS** prefer editing existing files
- **NEVER** proactively create documentation files (*.md, README)
- Only create what is explicitly requested

### Development Philosophy
- Educational clarity over production optimization
- Progressive learning approach (basic → advanced)
- Consider diverse student backgrounds
- Include detailed explanations for M.Tech level

## Course-Specific Considerations

### Tutorial Progression (T1-T15)
- T1-T3: TensorFlow basics, tensors, operations
- T4-T6: Neural networks, Keras, gradient descent
- T7-T9: Image processing, segmentation, features
- T10-T12: CNNs, data augmentation, LSTM
- T13-T15: Transfer learning, object detection

### Assessment Alignment
- Unit Test 1: Modules 1-2 (Sep 19)
- Unit Test 2: Modules 3-4 (Oct 31)
- Continuous lab assessment throughout
- Final exam: Comprehensive (Nov 21)

### Theory-Practice Balance
- Maintain 40-60% split (theory-practice)
- Every concept needs hands-on implementation
- Use industry-standard tools (TensorFlow, OpenCV)
- Google Colab recommended for students

## Known Issues & Workarounds

### Technical Issues
- Gradio incompatible with Python 3.13 (audioop module)
- Tkinter needs system-level installation on macOS
- TensorFlow protobuf warnings are non-critical
- Models auto-train if .keras file missing

### Environment Management
- Multiple virtual environments for different tasks
- Always activate correct environment before running
- Check dependencies match tutorial requirements

## Communication Style
- Be concise and direct
- Minimize output tokens
- Answer in 1-3 sentences when possible
- No unnecessary preambles or explanations
- One-word answers when appropriate

## When in Doubt
- Ask user for clarification
- Don't make assumptions about requirements
- Check existing files before creating new ones
- Verify commands with user if unknown
- Suggest saving important commands to CLAUDE.md