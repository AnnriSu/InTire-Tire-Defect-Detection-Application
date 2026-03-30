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
    const footer = document.querySelector('.bottom-nav'); // <-- grab footer

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
        if (footer) footer.style.display = 'none'; // <-- hide footer
        document.body.style.overflow = 'hidden'; // <-- disable scroll
    });

    overlay.addEventListener('click', () => {
        drawer.classList.remove('active');
        overlay.classList.remove('active');
        if (footer) footer.style.display = ''; // <-- restore footer
        document.body.style.overflow = ''; // <-- restore scroll
    });

    // Navigation helper
    window.navigateTo = (url) => window.location.href = url;

    // Logout helper
    window.logoutUser = () => {
        alert("Logged out successfully!");
        window.location.href = 'index.html';
    };
});