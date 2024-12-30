fn create_ml_project(cx: &mut zed::CommandContext) -> zed::Result<()> {
    let workspace = cx.workspace()?;
    let root_dir = workspace.root_directory()?;

    let directories = vec!["data", "models", "src", "tests"];
    for dir in directories {
        let dir_path = root_dir.join(dir);
        std::fs::create_dir_all(dir_path)?;
    }

    let files = vec![
        ("src/main.py", include_str!("../templates/main.py")),
        ("src/data_loader.py", include_str!("../templates/data_loader.py")),
        ("src/model.py", include_str!("../templates/model.py")),
        ("src/train.py", include_str!("../templates/train.py")),
        ("src/utils.py", include_str!("../templates/utils.py")),
        ("tests/test_model.py", include_str!("../templates/test_model.py")),
        ("requirements.txt", include_str!("../templates/requirements.txt")),
        ("README.md", include_str!("../templates/README.md")),
        (".gitignore", include_str!("../templates/.gitignore")),
    ];

    for (file_path, content) in files {
        let full_path = root_dir.join(file_path);
        std::fs::write(full_path, content)?;
    }

    cx.show_info_message("ML project created successfully!");
    Ok(())
}

