// drawer.js
const currentUser = {
    name: "User",
    email: "user@gmail.com"
};

// Load drawer.html and initialize
fetch('drawer.html')
  .then(res => res.text())
  .then(data => {
    const container = document.createElement('div');
    container.innerHTML = data;
    document.body.prepend(container);

    // Initialize drawer
    const usernameEl = document.getElementById('drawer-username');
    const emailEl = document.getElementById('drawer-email');
    const drawer = document.getElementById('drawer');
    const overlay = document.getElementById('overlay');

    // Update profile dynamically
    usernameEl.textContent = currentUser.name;
    emailEl.textContent = currentUser.email;

    // Toggle button
    let menuBtn = document.querySelector('.menu-toggle');
    if (!menuBtn) {
        menuBtn = document.createElement('button');
        menuBtn.className = 'menu-toggle';
        menuBtn.innerHTML = '<i class="fas fa-bars"></i>';
        document.body.prepend(menuBtn);
    }

    menuBtn.addEventListener('click', () => {
        drawer.classList.add('active');
        overlay.classList.add('active');
    });

    overlay.addEventListener('click', () => {
        drawer.classList.remove('active');
        overlay.classList.remove('active');
    });

    // Navigation helper
    window.navigateTo = (url) => window.location.href = url;

    // Logout helper
    window.logoutUser = () => {
        alert("Logged out successfully!");
        window.location.href = 'login.html';
    };
});